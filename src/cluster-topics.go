package main

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strings"

	"github.com/jmoiron/sqlx"
	"github.com/lib/pq"
	"github.com/pgvector/pgvector-go"
)

type Topic struct {
	ID        int              `db:"id" json:"id,omitempty"`
	Name      string           `db:"name" json:"name,omitempty"`
	Embedding *pgvector.Vector `db:"embedding" json:"embedding,omitempty"`
}

type TopicCluster struct {
	ID        int              `db:"id" json:"id,omitempty"`
	Title     string           `db:"title" json:"title,omitempty"`
	Embedding *pgvector.Vector `db:"embedding" json:"embedding,omitempty"`
}

type TopicDataPoint struct {
	TopicID   int
	TopicName string
	coords    Coordinates
}

func (tdp TopicDataPoint) Coordinates() Coordinates {
	return tdp.coords
}

type TopicClusterDataPoint struct {
	ClusterID    int
	ClusterTitle string
	coords       Coordinates
}

type OldClusterDataPoint struct {
	Title       string
	Coordinates Coordinates
}

type OldClusterData struct {
	DataPoints []OldClusterDataPoint
	TitlesStr  string
}

func toCoordinates(vector *pgvector.Vector) Coordinates {
	coordinates := vector.Slice()
	coords := make(Coordinates, len(coordinates))
	for i, v := range coordinates {
		coords[i] = float64(v)
	}
	return coords
}

func toPGVector(coords *Coordinates) pgvector.Vector {
	floatSlice := make([]float32, len(*coords))
	for i, v := range *coords {
		floatSlice[i] = float32(v)
	}
	return pgvector.NewVector(floatSlice)
}

func doClustering(oldClusters []TopicCluster, topics []Topic, numClusters int, logger *Logger) ([]Cluster, error) {
	topicDataPoints := make([]DataPoint, len(topics))
	for i, topic := range topics {
		if topic.Embedding == nil {
			return nil, fmt.Errorf("topic ID %d has nil embedding", topic.ID)
		}

		topicDataPoints[i] = TopicDataPoint{
			TopicID:   topic.ID,
			TopicName: topic.Name,
			coords:    toCoordinates(topic.Embedding),
		}
	}

	var centroids []Coordinates
	ok := len(oldClusters) == numClusters

	if ok {
		for _, cluster := range oldClusters {
			if cluster.Embedding == nil {
				ok = false
				break
			}
		}
	}

	if ok {
		logger.Debug("Using previous cluster centroids as initial centroids")

		centroids = make([]Coordinates, numClusters)
		for i, cluster := range oldClusters {
			centroids[i] = toCoordinates(cluster.Embedding)
		}
	} else {
		logger.Debug("Cannot reuse previous cluster centroids due to nil embeddings or size mismatch; initializing new centroids using k-means++")

		centroids = kmeanspp(numClusters, topicDataPoints, rand.New(rand.NewSource(42)))
	}

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

func getOldClusterData(oldClusters []TopicCluster) OldClusterData {
	var dataPoints []OldClusterDataPoint
	var titles []string

	for _, cluster := range oldClusters {
		if cluster.Embedding != nil {
			dataPoints = append(dataPoints, OldClusterDataPoint{
				Title:       cluster.Title,
				Coordinates: toCoordinates(cluster.Embedding),
			})
		}
		titles = append(titles, cluster.Title)
	}

	titlesStr := strings.Join(titles, ", ")

	return OldClusterData{
		DataPoints: dataPoints,
		TitlesStr:  titlesStr,
	}
}

func decideClusterTitle(cluster Cluster, closestTopics []DataPointDistance, oldClusterData OldClusterData, logger *Logger) (string, error) {
	for _, previousDataPoint := range oldClusterData.DataPoints {
		if distance(cluster.Centroid, previousDataPoint.Coordinates) < 0.01 {
			logger.Debug("Reusing title from previous topic clusters: " + previousDataPoint.Title)
			return previousDataPoint.Title, nil
		}
	}

	title, err := getClusterTitle(closestTopics, oldClusterData.TitlesStr, logger)
	if err != nil {
		return "", fmt.Errorf("failed to get cluster title: %w", err)
	}
	return title, nil
}

func clusterTopics(num_clusters int, num_closest_topics int, db *sqlx.DB, logger *Logger) error {
	tx, err := db.Beginx()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	logger.Debug("Retrieving previous topic clusters from database")

	var previousTopicClusters []TopicCluster
	err = tx.Select(&previousTopicClusters, "SELECT id, title, embedding FROM topic_clusters")
	if err != nil {
		return fmt.Errorf("failed to fetch existing topic clusters: %w", err)
	}

	oldClusterData := getOldClusterData(previousTopicClusters)

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

	// Sort for stability
	slices.SortFunc(topics, func(a, b Topic) int {
		return a.ID - b.ID
	})

	logger.Debug(fmt.Sprintf("Fetched %d topics from database", len(topics)))

	resultClusters, err := doClustering(previousTopicClusters, topics, num_clusters, logger)
	if err != nil {
		return fmt.Errorf("clustering failed: %w", err)
	}

	for i, cluster := range resultClusters {
		logger.Debug(fmt.Sprintf("=== Cluster %d (contains %d topics) ===", i, len(cluster.Points)))

		closestTopics := getClosestTopics(cluster, num_closest_topics)

		logger.Debug(fmt.Sprintf("Found %d closest topics for cluster %d", len(closestTopics), i))

		title, err := decideClusterTitle(cluster, closestTopics, oldClusterData, logger)
		if err != nil {
			return fmt.Errorf("failed to decide cluster title: %w", err)
		}

		logger.Debug("Inserting cluster into database")

		var clusterId int
		err = tx.Get(&clusterId, "INSERT INTO topic_clusters (title, embedding) VALUES ($1, $2) RETURNING id", title, toPGVector(&cluster.Centroid))
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
