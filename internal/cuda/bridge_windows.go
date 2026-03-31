//go:build windows && cuda

package cuda

import (
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

var (
	loadOnce sync.Once
	loadErr  error

	procAvailable *syscall.LazyProc
	procMalloc    *syscall.LazyProc
	procFree      *syscall.LazyProc
	procCopyH2D   *syscall.LazyProc
	procCopyD2H   *syscall.LazyProc
	procAdd       *syscall.LazyProc
	procReLU      *syscall.LazyProc
	procMatMul    *syscall.LazyProc
	procLastError *syscall.LazyProc
)

func Available() bool {
	if err := ensureLoaded(); err != nil {
		return false
	}
	return callInt(procAvailable) == 1
}

func MallocFloat32(count int) (unsafe.Pointer, error) {
	if err := ensureLoaded(); err != nil {
		return nil, err
	}
	var ptr uintptr
	code := callInt(procMalloc, uintptr(unsafe.Pointer(&ptr)), uintptr(count*4))
	if code != 0 {
		return nil, errorFromCode("cuda malloc", code)
	}
	return unsafe.Pointer(ptr), nil
}

func Free(ptr unsafe.Pointer) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	code := callInt(procFree, uintptr(ptr))
	if code != 0 {
		return errorFromCode("cuda free", code)
	}
	return nil
}

func CopyFloat32HostToDevice(dst unsafe.Pointer, src []float32) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	if len(src) == 0 {
		return nil
	}
	code := callInt(procCopyH2D, uintptr(dst), uintptr(unsafe.Pointer(&src[0])), uintptr(len(src)*4))
	if code != 0 {
		return errorFromCode("cuda memcpy h2d", code)
	}
	return nil
}

func CopyFloat32DeviceToHost(src unsafe.Pointer, dst []float32) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	if len(dst) == 0 {
		return nil
	}
	code := callInt(procCopyD2H, uintptr(unsafe.Pointer(&dst[0])), uintptr(src), uintptr(len(dst)*4))
	if code != 0 {
		return errorFromCode("cuda memcpy d2h", code)
	}
	return nil
}

func LaunchAddFloat32(a, b, out unsafe.Pointer, n int) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	code := callInt(procAdd, uintptr(a), uintptr(b), uintptr(out), uintptr(n))
	if code != 0 {
		return errorFromCode("cuda add", code)
	}
	return nil
}

func LaunchReLUFloat32(a, out unsafe.Pointer, n int) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	code := callInt(procReLU, uintptr(a), uintptr(out), uintptr(n))
	if code != 0 {
		return errorFromCode("cuda relu", code)
	}
	return nil
}

func LaunchMatMulFloat32(a, b, out unsafe.Pointer, m, k, n int) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	code := callInt(procMatMul, uintptr(a), uintptr(b), uintptr(out), uintptr(m), uintptr(k), uintptr(n))
	if code != 0 {
		return errorFromCode("cuda matmul", code)
	}
	return nil
}

func ensureLoaded() error {
	loadOnce.Do(func() {
		dllPath, err := cudaDLLPath()
		if err != nil {
			loadErr = err
			return
		}

		dll := syscall.NewLazyDLL(dllPath)
		procAvailable = dll.NewProc("tensor_cuda_available")
		procMalloc = dll.NewProc("tensor_cuda_malloc")
		procFree = dll.NewProc("tensor_cuda_free")
		procCopyH2D = dll.NewProc("tensor_cuda_memcpy_h2d")
		procCopyD2H = dll.NewProc("tensor_cuda_memcpy_d2h")
		procAdd = dll.NewProc("tensor_cuda_add_f32")
		procReLU = dll.NewProc("tensor_cuda_relu_f32")
		procMatMul = dll.NewProc("tensor_cuda_matmul_f32")
		procLastError = dll.NewProc("tensor_cuda_last_error")

		if err := dll.Load(); err != nil {
			loadErr = fmt.Errorf("load cuda dll: %w", err)
			return
		}
		for _, proc := range []*syscall.LazyProc{
			procAvailable,
			procMalloc,
			procFree,
			procCopyH2D,
			procCopyD2H,
			procAdd,
			procReLU,
			procMatMul,
			procLastError,
		} {
			if err := proc.Find(); err != nil {
				loadErr = fmt.Errorf("resolve %s: %w", proc.Name, err)
				return
			}
		}
	})
	return loadErr
}

func cudaDLLPath() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", errors.New("locate cuda bridge source")
	}
	rootDir := filepath.Dir(filepath.Dir(filepath.Dir(file)))
	return filepath.Join(rootDir, "cuda", "cudatensor.dll"), nil
}

func callInt(proc *syscall.LazyProc, args ...uintptr) int {
	r1, _, _ := proc.Call(args...)
	return int(r1)
}

func errorFromCode(op string, code int) error {
	msg := lastErrorString()
	if msg == "" {
		msg = fmt.Sprintf("code=%d", code)
	}
	return fmt.Errorf("%s failed: %s", op, msg)
}

func lastErrorString() string {
	if procLastError == nil {
		return ""
	}
	ptr, _, _ := procLastError.Call()
	if ptr == 0 {
		return ""
	}
	var buf []byte
	for offset := uintptr(0); ; offset++ {
		b := *(*byte)(unsafe.Pointer(ptr + offset))
		if b == 0 {
			break
		}
		buf = append(buf, b)
	}
	return string(buf)
}
