package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
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
- Der Thementitel soll abstrakter sein als die Einzeltitel, aber kein Sammelverzeichnis und kein zu allgemeiner Begriff.
- Führe keine neuen Inhalte oder politischen Konzepte ein, die nicht in den Titeln angelegt sind.
`

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func getClusterTitle(topicDistances []DataPointDistance, previousTitlesStr string, logger *Logger) (string, error) {
	var closestTopicsString string
	for _, td := range topicDistances {
		closestTopicsString = closestTopicsString + fmt.Sprintf("- %v (%.4f)\n", td.DataPoint.(TopicDataPoint).TopicName, td.Distance)
	}

	logger.Debug(fmt.Sprintf("Closest topics:\n%v", closestTopicsString))

	geminiClient, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  os.Getenv("GEMINI_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %w", err)
	}

	if previousTitlesStr == "" {
		previousTitlesStr = "(keine vorherigen Thementitel vorhanden)"
	}

	queryContent := []*genai.Content{
		{
			Parts: []*genai.Part{
				{
					Text: "Hier sind einige Thementitel von vorherigen Themenclustern. Verwende sie, um Überschneidungen zu vermeiden:",
				},
				{
					Text: previousTitlesStr,
				},
				{
					Text: "Die Themen, zu denen du einen passenden Thementitel finden sollst, lauten wie folgt:",
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

	var resp *genai.GenerateContentResponse
	var retriesLeft int = 3

	for retriesLeft > 0 {
		resp, err = geminiClient.Models.GenerateContent(context.Background(), GEMINI_MODEL, queryContent, contentConfig)
		retriesLeft--

		if err != nil {
			logger.Warn(fmt.Sprintf("Failed to generate content from Gemini (trying %d more times): %v", retriesLeft, err))
			continue
		}

		if strings.Trim(resp.Text(), " '\"") != "" && len(strings.Split(resp.Text(), " ")) <= 6 {
			break
		}

		logger.Warn(fmt.Sprintf("Received empty or too long title from Gemini, retrying... (Title: '%s')", truncateString(resp.Text(), 50)))
	}

	if retriesLeft == 0 {
		return "", fmt.Errorf("failed to generate a valid title from Gemini after multiple attempts: %w", err)
	}

	return strings.Trim(resp.Text(), " '\""), nil
}
