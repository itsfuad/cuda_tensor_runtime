package tensor

import (
	"sync"
	"time"
)

type ExecutionObservation struct {
	Op      OpKind
	Work    int
	Backend ExecBackend
	Elapsed time.Duration
}

var (
	observerMu        sync.RWMutex
	executionObserver func(ExecutionObservation)
)

func SetExecutionObserver(observer func(ExecutionObservation)) {
	observerMu.Lock()
	defer observerMu.Unlock()
	executionObserver = observer
}

func emitExecutionObservation(observation ExecutionObservation) {
	observerMu.RLock()
	observer := executionObserver
	observerMu.RUnlock()
	if observer != nil {
		observer(observation)
	}
}
