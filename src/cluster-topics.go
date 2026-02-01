package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/jmoiron/sqlx"
	"github.com/lib/pq"
	"github.com/pgvector/pgvector-go"
)

type Topic struct {
	ID        int             `db:"id" json:"id,omitempty"`
	Name      string          `db:"name" json:"name,omitempty"`
	Embedding pgvector.Vector `db:"embedding" json:"embedding,omitempty"`
}

type TopicDataPoint struct {
	TopicID   int
	TopicName string
	coords    Coordinates
}

func (tdp TopicDataPoint) Coordinates() Coordinates {
	return tdp.coords
}

func toCoordinates(coordinates []float32) Coordinates {
	coords := make(Coordinates, len(coordinates))
	for i, v := range coordinates {
		coords[i] = float64(v)
	}
	return coords
}

func doClustering(topics []Topic, numClusters int) ([]Cluster, error) {
	topicDataPoints := make([]DataPoint, len(topics))
	for i, topic := range topics {
		topicDataPoints[i] = TopicDataPoint{
			TopicID:   topic.ID,
			TopicName: topic.Name,
			coords:    toCoordinates(topic.Embedding.Slice()),
		}
	}

	centroids := kmeanspp(numClusters, topicDataPoints, rand.New(rand.NewSource(42)))
	clusters := kmeans_deterministic(numClusters, topicDataPoints, centroids, 200, rand.New(rand.NewSource(42)))

	return clusters, nil
}

func getClosestTopics(cluster Cluster, n int) []DataPointDistance {
	distances := make([]DataPointDistance, len(cluster.Points))
	for j, dataPoint := range cluster.Points {
		distances[j] = DataPointDistance{DataPoint: dataPoint, Distance: distance(dataPoint.Coordinates(), cluster.Centroid)}
	}

	sort.Slice(distances, func(i int, j int) bool {
		return distances[i].Distance < distances[j].Distance
	})

	return distances[:int(math.Min(float64(n), float64(len(distances))))]
}

func clusterTopics(num_clusters int, num_closest_topics int, db *sqlx.DB, logger *Logger) error {
	tx, err := db.Beginx()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	logger.Debug("Clearing and resetting topic_clusters table")

	_, err = tx.Exec("DELETE FROM topic_clusters")
	if err != nil {
		return fmt.Errorf("failed to clear topic_clusters table: %w", err)
	}

	_, err = tx.Exec("ALTER SEQUENCE topic_clusters_id_seq RESTART WITH 1")
	if err != nil {
		return fmt.Errorf("failed to reset topic_clusters sequence: %w", err)
	}

	logger.Debug("Fetching topics from database")

	var topics []Topic
	err = tx.Select(&topics, "SELECT id, name, embedding FROM topics")
	if err != nil {
		return fmt.Errorf("failed to fetch topics: %w", err)
	}

	logger.Debug(fmt.Sprintf("Fetched %d topics from database", len(topics)))

	resultClusters, err := doClustering(topics, num_clusters)
	if err != nil {
		return fmt.Errorf("clustering failed: %w", err)
	}

	for i, cluster := range resultClusters {
		logger.Debug(fmt.Sprintf("=== Cluster %d (contains %d topics) ===", i, len(cluster.Points)))

		closestTopics := getClosestTopics(cluster, num_closest_topics)

		logger.Debug(fmt.Sprintf("Found %d closest topics for cluster %d", len(closestTopics), i))

		title, err := getClusterTitle(closestTopics, logger)
		if err != nil {
			return fmt.Errorf("failed to get cluster title: %w", err)
		}

		logger.Debug("Inserting cluster into database")

		var clusterId int
		err = tx.Get(&clusterId, "INSERT INTO topic_clusters (title) VALUES ($1) RETURNING id", title)
		if err != nil {
			return fmt.Errorf("failed to insert topic cluster: %w", err)
		}

		var clusterTopicIds []int
		for _, dataPoint := range cluster.Points {
			topicDataPoint := dataPoint.(TopicDataPoint)
			clusterTopicIds = append(clusterTopicIds, topicDataPoint.TopicID)
		}

		logger.Debug(fmt.Sprintf("Assigning %d topics to cluster ID %d", len(clusterTopicIds), clusterId))

		_, err = tx.Exec("UPDATE topics SET cluster_id = $1 WHERE id = ANY($2)", clusterId, pq.Array(clusterTopicIds))
		if err != nil {
			return fmt.Errorf("failed to update topics with cluster ID: %w", err)
		}

		logger.Info(fmt.Sprintf("Assigned %d topics to cluster %v (id: %d)", len(clusterTopicIds), title, clusterId))
	}

	err = tx.Commit()
	if err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	logger.Info("Topic clustering completed successfully.")
	return nil
}
