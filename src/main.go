package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/jmoiron/sqlx"
	"github.com/joho/godotenv"
	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
	"github.com/pgvector/pgvector-go"
	"google.golang.org/genai"

	"github.com/lib/pq"
	_ "github.com/lib/pq"
)

const GEMINI_MODEL string = "gemini-2.5-flash"

const SYSTEM_INSTRUCTION string = `
Du erhältst einige Thementitel, die mithilfe eines k-means-Algorithmus aus Redebeiträgen des Deutschen Bundestages extrahiert wurden.
Zu jedem Thementitel ist ein Distanzwert angegeben, der seine Nähe zum Clusterzentrum beschreibt (kleiner = zentraler).

Aufgabe:
Bestimme einen übergeordneten Thementitel, der alle genannten Themen inhaltlich subsumiert.
Gib nur den übergeordneten Thementitel zurück, ohne weitere Erklärungen oder Formatierungen.

Vorgaben:
- Verwende eine kurze nominale Wortgruppe (1-6 Wörter).
- Der Titel soll abstrakter sein als die Einzeltitel, aber kein Sammelverzeichnis und kein zu allgemeiner Begriff.
- Keine neuen Inhalte oder politischen Konzepte einführen, die nicht in den Titeln angelegt sind.
`

var serviceLogPrefix = "TopicClusterer"

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

func buildDatabaseURL() string {
	host := os.Getenv("DATABASE_HOST")
	if host == "" {
		log.Fatal("DATABASE_HOST is required")
	}

	port := os.Getenv("DATABASE_PORT")
	if port == "" {
		log.Fatal("DATABASE_PORT is required")
	}

	user := os.Getenv("DATABASE_USER")
	if user == "" {
		log.Fatal("DATABASE_USER is required")
	}

	password := os.Getenv("DATABASE_PASSWORD")
	if password == "" {
		log.Fatal("DATABASE_PASSWORD is required")
	}

	dbname := os.Getenv("DATABASE_NAME")
	if dbname == "" {
		log.Fatal("DATABASE_NAME is required")
	}

	connStr := fmt.Sprintf("postgres://%s:%s@%s:%s/%s",
		user, password, host, port, dbname)

	if sslmode := os.Getenv("DATABASE_SSLMODE"); sslmode != "" {
		connStr += fmt.Sprintf("?sslmode=%s", sslmode)
	}

	return connStr
}

func clusterTopics(topics []Topic, numClusters int) (clusters.Clusters, error) {
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

func getClusterTitle(topicDistances []TopicDistance) (string, error) {
	var closestTopicsString string
	for _, td := range topicDistances {
		closestTopicsString = closestTopicsString + fmt.Sprintf("- %v (%.4f)\n", td.obs.TopicName, td.distance)
	}

	fmt.Println(closestTopicsString)

	geminiClient, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  os.Getenv("GEMINI_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %w", err)
	}

	queryContent := []*genai.Content{
		{
			Parts: []*genai.Part{
				{
					Text: "The topics are as follows:",
				},
				{
					Text: closestTopicsString,
				},
			},
		},
	}

	contentConfig := &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{
				{Text: SYSTEM_INSTRUCTION},
			},
		},
		ResponseMIMEType: "text/plain",
	}

	resp, err := geminiClient.Models.GenerateContent(context.Background(), GEMINI_MODEL, queryContent, contentConfig)
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	title := resp.Text()
	return title, nil
}

func main() {
	_ = godotenv.Load()

	LOG_LEVEL, err := GetLogLevel(os.Getenv("LOG_LEVEL"))
	if err != nil {
		log.Fatalf("Invalid LOG_LEVEL: %v", err)
	}

	dbURL := buildDatabaseURL()

	var db *sqlx.DB
	attempt := 0
	for true {
		attempt++
		db, err = sqlx.Connect("postgres", dbURL)
		if err == nil {
			defer db.Close()
			break
		}

		log.Printf(
			"Failed to connect to database (attempt=%d host=%q port=%q db=%q sslmode=%q): %v",
			attempt,
			os.Getenv("DATABASE_HOST"),
			os.Getenv("DATABASE_PORT"),
			os.Getenv("DATABASE_NAME"),
			os.Getenv("DATABASE_SSLMODE"),
			err,
		)
		time.Sleep(time.Second)
	}

	logger := NewLogger(db, &LOG_LEVEL, &LOG_LEVEL, serviceLogPrefix)

	NUM_CLUSTERS, err := strconv.Atoi(os.Getenv("NUM_CLUSTERS"))
	if err != nil {
		logger.Error("NUM_CLUSTERS is required and must be an integer")
		return
	}

	NUM_CLOSEST_TOPICS, err := strconv.Atoi(os.Getenv("NUM_CLOSEST_TOPICS"))
	if err != nil {
		logger.Error("NUM_CLOSEST_TOPICS is required and must be an integer")
		return
	}

	tx, err := db.Beginx()
	if err != nil {
		logger.Error(fmt.Sprintf("Failed to begin transaction: %v", err))
		return
	}
	defer tx.Rollback()

	_, err = tx.Exec("DELETE FROM topic_clusters")
	if err != nil {
		logger.Error(fmt.Sprintf("Failed to clear topic_clusters table: %v", err))
		return
	}

	_, err = tx.Exec("ALTER SEQUENCE topic_clusters_id_seq RESTART WITH 1")
	if err != nil {
		logger.Error(fmt.Sprintf("Failed to reset topic_clusters sequence: %v", err))
		return
	}

	var topics []Topic
	err = tx.Select(&topics, "SELECT id, name, embedding FROM topics")
	if err != nil {
		logger.Error(fmt.Sprintf("Failed to fetch topics: %v", err))
		return
	}

	logger.Debug(fmt.Sprintf("Fetched %d topics from database\n", len(topics)))

	resultClusters, err := clusterTopics(topics, NUM_CLUSTERS)
	if err != nil {
		logger.Error(fmt.Sprintf("Clustering failed: %v", err))
		return
	}

	for i, cluster := range resultClusters {
		logger.Debug(fmt.Sprintf("\n=== Cluster %d (contains %d topics) ===\n", i, len(cluster.Observations)))

		closestTopics := getClosestTopics(cluster, NUM_CLOSEST_TOPICS)

		title, err := getClusterTitle(closestTopics)
		if err != nil {
			logger.Error(fmt.Sprintf("Failed to get cluster title: %v", err))
			return
		}

		logger.Info(fmt.Sprintf("Cluster Title: %s", title))

		var clusterId int
		err = tx.Get(&clusterId, "INSERT INTO topic_clusters (title) VALUES ($1) RETURNING id", title)
		if err != nil {
			logger.Error(fmt.Sprintf("Failed to insert topic cluster: %v", err))
			return
		}

		var clusterTopicIds []int
		for _, obs := range cluster.Observations {
			topicObs := obs.(TopicObservation)
			clusterTopicIds = append(clusterTopicIds, topicObs.TopicID)
		}

		_, err = tx.Exec("UPDATE topics SET cluster_id = $1 WHERE id = ANY($2)", clusterId, pq.Array(clusterTopicIds))
		if err != nil {
			logger.Error(fmt.Sprintf("Failed to update topics with cluster ID: %v", err))
			return
		}

		logger.Info(fmt.Sprintf("Assigned %d topics to cluster ID %d", len(clusterTopicIds), clusterId))
	}

	err = tx.Commit()
	if err != nil {
		logger.Error(fmt.Sprintf("Failed to commit transaction: %v", err))
	}

	logger.Info("Topic clustering completed successfully.")
}
