package benchfmt

import (
	"encoding/json"
	"os"
)

type Result struct {
	Name       string  `json:"name"`
	Size       int     `json:"size"`
	Device     string  `json:"device"`
	Planner    string  `json:"planner,omitempty"`
	Iterations int     `json:"iterations"`
	TotalMs    float64 `json:"total_ms"`
	AvgMs      float64 `json:"avg_ms"`
}

type Sample struct {
	Name      string  `json:"name"`
	Size      int     `json:"size"`
	Planner   string  `json:"planner"`
	Iteration int     `json:"iteration"`
	Backend   string  `json:"backend"`
	ElapsedMs float64 `json:"elapsed_ms"`
}

func WriteFile(path string, results []Result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(results)
}

func ReadFile(path string) ([]Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var results []Result
	if err := json.NewDecoder(f).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

func WriteSamples(path string, samples []Sample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(samples)
}

func ReadSamples(path string) ([]Sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var samples []Sample
	if err := json.NewDecoder(f).Decode(&samples); err != nil {
		return nil, err
	}
	return samples, nil
}
