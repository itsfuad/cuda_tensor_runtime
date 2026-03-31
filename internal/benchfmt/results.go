package benchfmt

import (
	"encoding/json"
	"os"
)

type Result struct {
	Name       string  `json:"name"`
	Size       int     `json:"size"`
	Device     string  `json:"device"`
	Iterations int     `json:"iterations"`
	TotalMs    float64 `json:"total_ms"`
	AvgMs      float64 `json:"avg_ms"`
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
