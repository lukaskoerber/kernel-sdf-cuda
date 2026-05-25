// omega_linear: CK-PCA Omega for the LINEAR kernel via the F F' shortcut.
//
// For the linear kernel, the reference's per-(t,s) block double-centering
//   K_tilde[i,j] = K[i,j] - row_means[i] - col_means[j] + all_mean,  K = Z_t Z_s'
// reduces algebraically to
//   K_tilde = (Z_t - zbar_t) (Z_s - zbar_s)',     zbar_t = (1/N_t) sum_i Z_t[i,:]
// so
//   Omega[t,s] = r_t' K_tilde r_s = F_t . F_s,
//   F_t = (Z_t - zbar_t)' r_t  in R^K, i.e. F[t,:] = Z_t' r_t - zbar_t * (sum r_t).
// Omega = F F' is then a single Dgemm.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "prep_loader.hpp"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e__ = (call);                                                  \
    if (e__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                   cudaGetErrorName(e__), __FILE__, __LINE__,                  \
                   cudaGetErrorString(e__));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t s__ = (call);                                               \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s__,             \
                   __FILE__, __LINE__);                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// One block per month t. blockDim.x threads stride over the K-axis (one or
// more characteristics per thread). Each thread scans the N_t rows once for
// its k(s); a single thread also reduces r_t to sum_r in shared memory.
__global__ void build_F_kernel(const double* __restrict__ Z,
                               const double* __restrict__ r,
                               const int64_t* __restrict__ offsets,
                               int64_t K,
                               double* __restrict__ F) {
  const int64_t t   = blockIdx.x;
  const int64_t off = offsets[t];
  const int64_t n   = offsets[t + 1] - off;

  __shared__ double sum_r_sh;
  if (threadIdx.x == 0) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += r[off + i];
    sum_r_sh = s;
  }
  __syncthreads();
  const double sum_r = sum_r_sh;
  const double inv_n = 1.0 / static_cast<double>(n);

  for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
    double sum_z  = 0.0;
    double sum_zr = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const double zik = Z[(off + i) * K + k];
      const double ri  = r[off + i];
      sum_z  += zik;
      sum_zr += zik * ri;
    }
    const double zbar = sum_z * inv_n;
    F[t * K + k] = sum_zr - zbar * sum_r;
  }
}

static void mkdir_p(const std::string& dir) {
  std::string cmd = "mkdir -p '" + dir + "'";
  if (std::system(cmd.c_str()) != 0)
    throw std::runtime_error("mkdir failed: " + dir);
}

int main(int argc, char** argv) {
  const std::string prep_dir = (argc > 1) ? argv[1] : "data/prep";
  const std::string out_dir  = (argc > 2) ? argv[2] : "data/out";
  const std::string out_path = out_dir + "/omega_linear.f64.bin";

  ksdf::PrepData p = ksdf::load_prep(prep_dir);
  const int64_t T = p.T, K = p.K, R = p.total_rows;
  std::printf("[omega-linear] T=%lld K=%lld total_rows=%lld\n",
              (long long)T, (long long)K, (long long)R);

  double  *dZ  = nullptr, *dR = nullptr, *dF = nullptr, *dOmega = nullptr;
  int64_t *dOff = nullptr;
  CUDA_CHECK(cudaMalloc(&dZ,    R * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dR,    R     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOff,  (T + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&dF,    T * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOmega, T * T * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(dZ,   p.Z,       R * K * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dR,   p.r,       R     * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dOff, p.offsets, (T+1) * sizeof(int64_t), cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  const int block = 128;
  build_F_kernel<<<static_cast<unsigned>(T), block>>>(dZ, dR, dOff, K, dF);
  CUDA_CHECK(cudaGetLastError());

  // F is row-major (T x K). Reinterpreted as col-major it is (K x T) with
  // leading dim K, i.e. cuBLAS sees M = F^T. Omega = F F^T = M^T M.
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const double alpha = 1.0, beta = 0.0;
  CUBLAS_CHECK(cublasDgemm(handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           /*m=*/T, /*n=*/T, /*k=*/K,
                           &alpha,
                           dF, /*lda=*/K,
                           dF, /*ldb=*/K,
                           &beta,
                           dOmega, /*ldc=*/T));

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  std::printf("[omega-linear] build_F + Dgemm = %.3f ms\n", ms);

  std::vector<double> h_omega(static_cast<size_t>(T) * T);
  CUDA_CHECK(cudaMemcpy(h_omega.data(), dOmega,
                       T * T * sizeof(double), cudaMemcpyDeviceToHost));

  mkdir_p(out_dir);
  std::ofstream fout(out_path, std::ios::binary);
  if (!fout) throw std::runtime_error("open(" + out_path + ") failed");
  fout.write(reinterpret_cast<const char*>(h_omega.data()),
             static_cast<std::streamsize>(h_omega.size() * sizeof(double)));
  if (!fout) throw std::runtime_error("write(" + out_path + ") failed");
  std::printf("[omega-linear] wrote %s (%zu bytes)\n",
              out_path.c_str(), h_omega.size() * sizeof(double));

  cublasDestroy(handle);
  cudaFree(dOmega);
  cudaFree(dF);
  cudaFree(dOff);
  cudaFree(dR);
  cudaFree(dZ);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  ksdf::free_prep(p);
  return 0;
}
