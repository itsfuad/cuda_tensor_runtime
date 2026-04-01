package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"sort"

	"encoding/json"
	"os"

	"github.com/itsfuad/cuda_tensor_runtime/internal/benchfmt"
)

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

	baseResults, err := benchfmt.ReadFile(*baseFlag)
	if err != nil {
		log.Fatal(err)
	}
	candidateResults, err := benchfmt.ReadFile(*candidateFlag)
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

func compare(baseResults, candidateResults []benchfmt.Result) ([]comparisonRow, error) {
	baseIndex := make(map[string]benchfmt.Result, len(baseResults))
	baseByNameSize := make(map[string][]benchfmt.Result, len(baseResults))
	for _, result := range baseResults {
		baseIndex[key(result)] = result
		baseByNameSize[nameSizeKey(result)] = append(baseByNameSize[nameSizeKey(result)], result)
	}

	rows := make([]comparisonRow, 0, len(candidateResults))
	for _, result := range candidateResults {
		base, ok := baseIndex[key(result)]
		if !ok {
			candidates := baseByNameSize[nameSizeKey(result)]
			if len(candidates) != 1 {
				return nil, fmt.Errorf("missing baseline result for workload=%s size=%d device=%s", result.Name, result.Size, result.Device)
			}
			base = candidates[0]
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

func key(result benchfmt.Result) string {
	return fmt.Sprintf("%s|%d|%s", result.Name, result.Size, result.Device)
}

func nameSizeKey(result benchfmt.Result) string {
	return fmt.Sprintf("%s|%d", result.Name, result.Size)
}

func printText(rows []comparisonRow, baseName, candidateName string) {
	fmt.Printf("%-16s %-8s %-8s %-16s %-16s %-12s\n", "workload", "size", "device", baseName+"_avg_ms", candidateName+"_avg_ms", "speedup")
	for _, row := range rows {
		speedup := fmt.Sprintf("%.3fx", row.Speedup)
		fmt.Printf("%-16s %-8d %-8s %-16.3f %-16.3f %-12s\n",
			row.Name, row.Size, row.Device, row.BaseAvgMs, row.CandidateAvgMs, speedup)
	}
}
