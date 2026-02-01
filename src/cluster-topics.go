package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/jmoiron/sqlx"
	"github.com/lib/pq"
	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
	"github.com/pgvector/pgvector-go"
)

type Topic struct {
	ID        int             `db:"id" json:"id,omitempty"`
	Name      string          `db:"name" json:"name,omitempty"`
	Embedding pgvector.Vector `db:"embedding" json:"embedding,omitempty"`
}

type TopicObservation struct {
	TopicID   int
	TopicName string
	Coords    []float32
}

type TopicDistance struct {
	obs      TopicObservation
	distance float64
}

func (t TopicObservation) Coordinates() clusters.Coordinates {
	coords := make(clusters.Coordinates, len(t.Coords))
	for i, v := range t.Coords {
		coords[i] = float64(v)
	}
	return coords
}

func (t TopicObservation) Distance(point clusters.Coordinates) float64 {
	var sum float64
	coords := t.Coordinates()
	for i, v := range coords {
		diff := v - point[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func doClustering(topics []Topic, numClusters int) (clusters.Clusters, error) {
	observations := make(clusters.Observations, len(topics))
	for i, topic := range topics {
		observations[i] = TopicObservation{
			TopicID:   topic.ID,
			TopicName: topic.Name,
			Coords:    topic.Embedding.Slice(),
		}
	}

	clusterer, err := kmeans.NewWithOptions(0.01, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create clusterer: %w", err)
	}

	clusters, err := clusterer.Partition(observations, numClusters)
	if err != nil {
		return nil, fmt.Errorf("failed to partition: %w", err)
	}

	return clusters, nil
}

func getClosestTopics(cluster clusters.Cluster, n int) []TopicDistance {
	distances := make([]TopicDistance, len(cluster.Observations))
	for j, obs := range cluster.Observations {
		topicObs := obs.(TopicObservation)
		distances[j] = TopicDistance{obs: topicObs, distance: topicObs.Distance(cluster.Center)}
	}

	sort.Slice(distances, func(i int, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	return distances[:int(math.Min(float64(n), float64(len(distances))))]
}

func clusterTopics(num_clusters int, num_closest_topics int, db *sqlx.DB, logger *Logger) error {
	tx, err := db.Beginx()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.Exec("DELETE FROM topic_clusters")
	if err != nil {
		return fmt.Errorf("failed to clear topic_clusters table: %w", err)
	}

	_, err = tx.Exec("ALTER SEQUENCE topic_clusters_id_seq RESTART WITH 1")
	if err != nil {
		return fmt.Errorf("failed to reset topic_clusters sequence: %w", err)
	}

	var topics []Topic
	err = tx.Select(&topics, "SELECT id, name, embedding FROM topics")
	if err != nil {
		return fmt.Errorf("failed to fetch topics: %w", err)
	}

	logger.Debug(fmt.Sprintf("Fetched %d topics from database\n", len(topics)))

	resultClusters, err := doClustering(topics, num_clusters)
	if err != nil {
		return fmt.Errorf("clustering failed: %w", err)
	}

	for i, cluster := range resultClusters {
		logger.Debug(fmt.Sprintf("\n=== Cluster %d (contains %d topics) ===\n", i, len(cluster.Observations)))

		closestTopics := getClosestTopics(cluster, num_closest_topics)

		title, err := getClusterTitle(closestTopics, logger)
		if err != nil {
			return fmt.Errorf("failed to get cluster title: %w", err)
		}

		logger.Info(fmt.Sprintf("Cluster Title: %s", title))

		var clusterId int
		err = tx.Get(&clusterId, "INSERT INTO topic_clusters (title) VALUES ($1) RETURNING id", title)
		if err != nil {
			return fmt.Errorf("failed to insert topic cluster: %w", err)
		}

		var clusterTopicIds []int
		for _, obs := range cluster.Observations {
			topicObs := obs.(TopicObservation)
			clusterTopicIds = append(clusterTopicIds, topicObs.TopicID)
		}

		_, err = tx.Exec("UPDATE topics SET cluster_id = $1 WHERE id = ANY($2)", clusterId, pq.Array(clusterTopicIds))
		if err != nil {
			return fmt.Errorf("failed to update topics with cluster ID: %w", err)
		}

		logger.Info(fmt.Sprintf("Assigned %d topics to cluster ID %d", len(clusterTopicIds), clusterId))
	}

	err = tx.Commit()
	if err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	logger.Info("Topic clustering completed successfully.")
	return nil
}
