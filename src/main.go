package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/jmoiron/sqlx"
	"github.com/joho/godotenv"
)

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

	err = clusterTopics(NUM_CLUSTERS, NUM_CLOSEST_TOPICS, db, logger)
	if err != nil {
		logger.Error(fmt.Sprintf("Failed to cluster topics: %v", err))
		return
	}
}
