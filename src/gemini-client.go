package main

import (
	"context"
	"fmt"
	"os"

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
- Der Titel soll abstrakter sein als die Einzeltitel, aber kein Sammelverzeichnis und kein zu allgemeiner Begriff.
- Keine neuen Inhalte oder politischen Konzepte einführen, die nicht in den Titeln angelegt sind.
`

func getClusterTitle(topicDistances []TopicDistance, logger *Logger) (string, error) {
	var closestTopicsString string
	for _, td := range topicDistances {
		closestTopicsString = closestTopicsString + fmt.Sprintf("- %v (%.4f)\n", td.obs.TopicName, td.distance)
	}

	logger.Debug(closestTopicsString)

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
