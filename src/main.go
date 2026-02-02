package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/jmoiron/sqlx"
	"github.com/joho/godotenv"
)

var topicsClustererRunning bool = false
var serviceLogPrefix = "TopicClusterer"

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

func tryAcquireTopicClusterLock(db *sqlx.DB, logger *Logger) (bool, error) {
	logger.Debug("Trying to acquire topic cluster lock")
	startTime := time.Now()
	result, err := db.Exec(`
		UPDATE locks
		SET locked = TRUE, heartbeat = CURRENT_TIMESTAMP
		WHERE name = 'topic-cluster'
			AND (locked = FALSE OR heartbeat < (CURRENT_TIMESTAMP - INTERVAL '3 minutes'))
	`)
	logger.Debug(fmt.Sprintf("Topic cluster lock acquisition query took %v", time.Since(startTime)))
	if err != nil {
		return false, fmt.Errorf("failed to run topic cluster lock acquisition query: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("failed to check rows affected for topic cluster lock: %w", err)
	}
	if rowsAffected == 0 {
		logger.Info("Another instance is already running topic clustering.")
		return false, nil
	}

	return true, nil
}

func topicClustererWorker(NUM_CLUSTERS int, NUM_CLOSEST_TOPICS int, TOPIC_CLUSTERING_SLEEP_HOURS int, db *sqlx.DB, logger *Logger) {
	for {
		for !topicsClustererRunning {
			logger.Info("Topic clustering is currently paused. Sleeping for 10 seconds.")
			time.Sleep(10 * time.Second)
		}

		logger.Info("Starting topic clustering run")

		err := clusterTopics(NUM_CLUSTERS, NUM_CLOSEST_TOPICS, db, logger)

		if err == nil {
			logger.Info("Topic clustering run completed successfully.")
			logger.Info(fmt.Sprintf("Sleeping for %d hours before next run.", TOPIC_CLUSTERING_SLEEP_HOURS))
			time.Sleep(time.Duration(TOPIC_CLUSTERING_SLEEP_HOURS) * time.Hour)
		} else {
			logger.Error(fmt.Sprintf("Failed to cluster topics: %v", err))
			logger.Info("Sleeping for 60 minutes before next attempt.")
			time.Sleep(60 * time.Minute)
		}
	}
}

func heartbeatWorker(db *sqlx.DB, logger *Logger) {
	for {
		_, err := db.Exec("UPDATE locks SET heartbeat = CURRENT_TIMESTAMP WHERE name = 'topic-cluster'")
		if err != nil {
			logger.Fatal(fmt.Sprintf("heartbeat update failed: %v", err))
		} else {
			logger.Debug("heartbeat update succeeded")
		}

		time.Sleep(1 * time.Minute)
	}
}

func attemptStartTopicClustering(db *sqlx.DB, logger *Logger, NUM_CLUSTERS int, NUM_CLOSEST_TOPICS int, TOPIC_CLUSTERING_SLEEP_HOURS int) {
	for {
		acquired, err := tryAcquireTopicClusterLock(db, logger)
		if err != nil {
			logger.Fatal(fmt.Sprintf("Failed to try to acquire topic cluster lock: %v", err))
		}

		if acquired {
			break
		}

		time.Sleep(1 * time.Minute)
	}

	logger.Info("Topic cluster lock acquired successfully.")

	go heartbeatWorker(db, logger)

	topicClustererWorker(NUM_CLUSTERS, NUM_CLOSEST_TOPICS, TOPIC_CLUSTERING_SLEEP_HOURS, db, logger)
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

	TOPIC_CLUSTERING_SLEEP_HOURS, err := strconv.Atoi(os.Getenv("TOPIC_CLUSTERING_SLEEP_HOURS"))
	if err != nil {
		logger.Error("TOPIC_CLUSTERING_SLEEP_HOURS is required and must be an integer")
		return
	}

	topicsClustererRunning, err = strconv.ParseBool(os.Getenv("BEGIN_CLUSTERING_ON_STARTUP"))
	if err != nil {
		logger.Error("BEGIN_CLUSTERING_ON_STARTUP is required and must be a boolean")
		return
	}

	go attemptStartTopicClustering(db, logger, NUM_CLUSTERS, NUM_CLOSEST_TOPICS, TOPIC_CLUSTERING_SLEEP_HOURS)

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "Server healthy")
	})

	http.HandleFunc("/control-clustering", func(w http.ResponseWriter, r *http.Request) {
		topicsClustererRunning = r.URL.Query().Get("start") == "true"
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "Clustering status updated with start="+strconv.FormatBool(topicsClustererRunning))
	})

	logger.Info("HTTP server starting on :8080 (endpoints: /health, /control-clustering)")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		logger.Fatal(fmt.Sprintf("HTTP server stopped unexpectedly: %v", err))
	}
}
