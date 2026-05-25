#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                    \
                   cudaGetErrorName(err__), __FILE__, __LINE__,               \
                   cudaGetErrorString(err__));                                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__global__ void axpy(double a, const double* x, double* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int rt = 0, drv = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  std::printf("device     : %s (sm_%d%d, %.1f GB)\n", p.name, p.major, p.minor,
              (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  std::printf("cuda rt/drv: %d.%d / %d.%d\n", rt / 1000, (rt % 100) / 10,
              drv / 1000, (drv % 100) / 10);

  constexpr int N = 1 << 16;
  double *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dx, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(double)));

  // host init: x = 1.0, y = 2.0; expect y' = 2*1 + 2 = 4 everywhere.
  double *hx = new double[N], *hy = new double[N];
  for (int i = 0; i < N; ++i) { hx[i] = 1.0; hy[i] = 2.0; }
  CUDA_CHECK(cudaMemcpy(dx, hx, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, hy, N * sizeof(double), cudaMemcpyHostToDevice));

  const int block = 256;
  const int grid = (N + block - 1) / block;
  axpy<<<grid, block>>>(2.0, dx, dy, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hy, dy, N * sizeof(double), cudaMemcpyDeviceToHost));
  double max_abs_err = 0.0;
  for (int i = 0; i < N; ++i) {
    double e = std::abs(hy[i] - 4.0);
    if (e > max_abs_err) max_abs_err = e;
  }
  std::printf("axpy N=%d, max|y-4| = %.3e\n", N, max_abs_err);

  delete[] hx; delete[] hy;
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return max_abs_err == 0.0 ? 0 : 2;
}
