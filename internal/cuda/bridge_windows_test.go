//go:build windows && cuda

package cuda

import "testing"

func TestFloat32RoundTripStress(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable")
	}

	sizes := []int{256, 4096, 65536}
	for _, size := range sizes {
		src := make([]float32, size)
		dst := make([]float32, size)
		for i := range src {
			src[i] = float32((i % 31) - 15)
		}

		for iter := 0; iter < 128; iter++ {
			buf, err := MallocFloat32(size)
			if err != nil {
				t.Fatalf("MallocFloat32(%d) on iter %d: %v", size, iter, err)
			}
			if err := CopyFloat32HostToDevice(buf, src); err != nil {
				_ = Free(buf)
				t.Fatalf("CopyFloat32HostToDevice(%d) on iter %d: %v", size, iter, err)
			}
			if err := CopyFloat32DeviceToHost(buf, dst); err != nil {
				_ = Free(buf)
				t.Fatalf("CopyFloat32DeviceToHost(%d) on iter %d: %v", size, iter, err)
			}
			if err := Free(buf); err != nil {
				t.Fatalf("Free(%d) on iter %d: %v", size, iter, err)
			}
			for i := range src {
				if dst[i] != src[i] {
					t.Fatalf("round-trip mismatch size=%d iter=%d index=%d got=%v want=%v", size, iter, i, dst[i], src[i])
				}
			}
		}
	}
}
