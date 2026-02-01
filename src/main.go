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

	NUM_CLUSTERS, err := strconv.Atoi(os.Getenv("NUM_CLUSTERS"))
	if err != nil {
		log.Fatal("NUM_CLUSTERS is required and must be an integer")
	}

	NUM_CLOSEST_TOPICS, err := strconv.Atoi(os.Getenv("NUM_CLOSEST_TOPICS"))
	if err != nil {
		log.Fatal("NUM_CLOSEST_TOPICS is required and must be an integer")
	}

	dbURL := buildDatabaseURL()

	var db *sqlx.DB
	for true {
		db, err = sqlx.Connect("postgres", dbURL)
		if err == nil {
			defer db.Close()
			break
		}
		log.Printf("Failed to connect to database: %v", err)
		time.Sleep(time.Second)
	}

	tx, err := db.Beginx()
	if err != nil {
		log.Fatalf("Failed to begin transaction: %v", err)
	}
	defer tx.Rollback()

	_, err = tx.Exec("DELETE FROM topic_clusters")
	if err != nil {
		log.Fatalf("Failed to clear topic_clusters table: %v", err)
	}

	_, err = tx.Exec("ALTER SEQUENCE topic_clusters_id_seq RESTART WITH 1")
	if err != nil {
		log.Fatalf("Failed to reset topic_clusters sequence: %v", err)
	}

	var topics []Topic
	err = tx.Select(&topics, "SELECT id, name, embedding FROM topics")
	if err != nil {
		log.Fatalf("Failed to fetch topics: %v", err)
	}

	fmt.Printf("Fetched %d topics from database\n", len(topics))

	resultClusters, err := clusterTopics(topics, NUM_CLUSTERS)
	if err != nil {
		log.Fatalf("Clustering failed: %v", err)
	}

	for i, cluster := range resultClusters {
		fmt.Printf("\n=== Cluster %d (contains %d topics) ===\n", i, len(cluster.Observations))

		closestTopics := getClosestTopics(cluster, NUM_CLOSEST_TOPICS)

		title, err := getClusterTitle(closestTopics)
		if err != nil {
			log.Printf("Failed to get cluster title: %v", err)
			continue
		}

		fmt.Printf("Cluster Title: %s\n", title)

		var clusterId int
		err = tx.Get(&clusterId, "INSERT INTO topic_clusters (title) VALUES ($1) RETURNING id", title)
		if err != nil {
			log.Fatalf("Failed to insert topic cluster: %v", err)
		}

		var clusterTopicIds []int
		for _, obs := range cluster.Observations {
			topicObs := obs.(TopicObservation)
			clusterTopicIds = append(clusterTopicIds, topicObs.TopicID)
		}

		_, err = tx.Exec("UPDATE topics SET cluster_id = $1 WHERE id = ANY($2)", clusterId, pq.Array(clusterTopicIds))
		if err != nil {
			log.Fatalf("Failed to update topics with cluster ID: %v", err)
		}

		fmt.Printf("Assigned %d topics to cluster ID %d\n", len(clusterTopicIds), clusterId)
	}

	err = tx.Commit()
	if err != nil {
		log.Fatalf("Failed to commit transaction: %v", err)
	}
}
