package main

import (
	"log"
	"math"
	"math/rand"
)

type Coordinates []float64

type DataPoint interface {
	Coordinates() Coordinates
}

type DataPointDistance struct {
	DataPoint DataPoint
	Distance  float64
}

type Cluster struct {
	Points   []DataPoint
	Centroid Coordinates
}

func distance(a, b Coordinates) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func getClosest(point DataPoint, centroids []Coordinates) (int, float64) {
	minDist := math.MaxFloat64
	closest := 0
	for i, centroid := range centroids {
		dist := distance(point.Coordinates(), centroid)
		if dist < minDist {
			minDist = dist
			closest = i
		}
	}
	return closest, minDist
}

func calculateCentroid(points []DataPoint, indices []int, rng *rand.Rand) Coordinates {
	centroid := make(Coordinates, len(points[0].Coordinates()))

	if len(indices) == 0 {
		log.Println("-------------------------- BAD --------------------------")
		return points[rng.Intn(len(points))].Coordinates()
	}

	for _, index := range indices {
		for i, value := range points[index].Coordinates() {
			centroid[i] += value
		}
	}
	for i := range centroid {
		centroid[i] /= float64(len(indices))
	}
	return centroid
}

func kmeans_deterministic(k int, data []DataPoint, centroids []Coordinates, maxIterations int, rng *rand.Rand) []Cluster {
	var clusterIndices [][]int

	for ; maxIterations > 0; maxIterations-- {
		clusterIndices = make([][]int, k)

		for i, point := range data {
			closest, _ := getClosest(point, centroids)
			clusterIndices[closest] = append(clusterIndices[closest], i)
		}

		newCentroids := make([]Coordinates, k)
		for i, cluster := range clusterIndices {
			newCentroids[i] = calculateCentroid(data, cluster, rng)
		}

		converged := true
		for i := range centroids {
			if distance(centroids[i], newCentroids[i]) > 1e-6 {
				converged = false
				break
			}
		}

		centroids = newCentroids

		if converged {
			break
		}
	}

	clusters := make([]Cluster, k)
	for i, indices := range clusterIndices {
		dataPoints := make([]DataPoint, len(indices))
		for j := range indices {
			dataPoints[j] = data[indices[j]]
		}
		clusters[i].Points = dataPoints
		clusters[i].Centroid = centroids[i]
	}
	return clusters
}

func kmeanspp(k int, data []DataPoint, rng *rand.Rand) []Coordinates {
	var centroids []Coordinates

	firstIndex := rng.Intn(len(data))
	centroids = append(centroids, data[firstIndex].Coordinates())

	for len(centroids) < k {
		squaredDistances := make([]float64, len(data))
		var totalSquaredDistance float64

		for i, point := range data {
			_, dist := getClosest(point, centroids)
			squaredDistances[i] = dist * dist
			totalSquaredDistance += dist * dist
		}

		r := rng.Float64() * totalSquaredDistance
		cumulative := 0.0
		for i, dist := range squaredDistances {
			cumulative += dist
			if cumulative >= r {
				centroids = append(centroids, data[i].Coordinates())
				break
			}
		}
	}
	return centroids
}
