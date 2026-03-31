package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
)

type benchResult struct {
	Name       string  `json:"name"`
	Size       int     `json:"size"`
	Device     string  `json:"device"`
	Iterations int     `json:"iterations"`
	TotalMs    float64 `json:"total_ms"`
	AvgMs      float64 `json:"avg_ms"`
}

type comparisonRow struct {
	Name           string
	Size           int
	Device         string
	BaseAvgMs      float64
	CandidateAvgMs float64
	Speedup        float64
}

func main() {
	baseFlag := flag.String("base", "", "path to baseline JSON results")
	candidateFlag := flag.String("candidate", "", "path to candidate JSON results")
	baseNameFlag := flag.String("base-name", "baseline", "label for the baseline results")
	candidateNameFlag := flag.String("candidate-name", "candidate", "label for the candidate results")
	formatFlag := flag.String("format", "text", "output format: text or json")
	flag.Parse()

	if *baseFlag == "" || *candidateFlag == "" {
		log.Fatal("both -base and -candidate are required")
	}

	baseResults, err := readResults(*baseFlag)
	if err != nil {
		log.Fatal(err)
	}
	candidateResults, err := readResults(*candidateFlag)
	if err != nil {
		log.Fatal(err)
	}

	rows, err := compare(baseResults, candidateResults)
	if err != nil {
		log.Fatal(err)
	}

	switch *formatFlag {
	case "text":
		printText(rows, *baseNameFlag, *candidateNameFlag)
	case "json":
		if err := json.NewEncoder(os.Stdout).Encode(rows); err != nil {
			log.Fatal(err)
		}
	default:
		log.Fatalf("unsupported format %q", *formatFlag)
	}
}

func readResults(path string) ([]benchResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var results []benchResult
	if err := json.NewDecoder(f).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

func compare(baseResults, candidateResults []benchResult) ([]comparisonRow, error) {
	baseIndex := make(map[string]benchResult, len(baseResults))
	for _, result := range baseResults {
		baseIndex[key(result)] = result
	}

	rows := make([]comparisonRow, 0, len(candidateResults))
	for _, result := range candidateResults {
		base, ok := baseIndex[key(result)]
		if !ok {
			return nil, fmt.Errorf("missing baseline result for workload=%s size=%d device=%s", result.Name, result.Size, result.Device)
		}
		speedup := math.NaN()
		if result.AvgMs != 0 {
			speedup = base.AvgMs / result.AvgMs
		}
		rows = append(rows, comparisonRow{
			Name:           result.Name,
			Size:           result.Size,
			Device:         result.Device,
			BaseAvgMs:      base.AvgMs,
			CandidateAvgMs: result.AvgMs,
			Speedup:        speedup,
		})
	}

	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Name != rows[j].Name {
			return rows[i].Name < rows[j].Name
		}
		if rows[i].Size != rows[j].Size {
			return rows[i].Size < rows[j].Size
		}
		return rows[i].Device < rows[j].Device
	})
	return rows, nil
}

func key(result benchResult) string {
	return fmt.Sprintf("%s|%d|%s", result.Name, result.Size, result.Device)
}

func printText(rows []comparisonRow, baseName, candidateName string) {
	fmt.Printf("%-16s %-8s %-8s %-16s %-16s %-12s\n", "workload", "size", "device", baseName+"_avg_ms", candidateName+"_avg_ms", "speedup")
	for _, row := range rows {
		speedup := fmt.Sprintf("%.3fx", row.Speedup)
		fmt.Printf("%-16s %-8d %-8s %-16.3f %-16.3f %-12s\n",
			row.Name, row.Size, row.Device, row.BaseAvgMs, row.CandidateAvgMs, speedup)
	}
}
