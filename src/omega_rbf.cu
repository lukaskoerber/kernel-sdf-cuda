// omega_rbf: CK-PCA Omega for the RBF kernel.
//
// For each pair (t, s) with t <= s and N1=N_t, N2=N_s:
//   1. Dgemm: M = Z_t @ Z_s' in row-major (n_t x n_s).
//   2. Fused reduction (single block, single launch): one pass over the
//      (i, j) grid computing K[i,j] = exp(-(z1n[i]+z2n[j]-2*M[i,j])/(2c^2))
//      on the fly, accumulating four scalars:
//        S_K  = sum K              S_K1 = sum r_t[i] * K
//        S_1K = sum K * r_s[j]     S_ab = sum r_t[i] * K * r_s[j]
//   3. Omega[t,s] = S_ab - S_K1*R_s/N_s - R_t*S_1K/N_t + S_K*R_t*R_s/(N_t*N_s)
//      (algebraically equivalent to the reference's
//          K_tilde = K - rm - cm + am;   r_t' K_tilde r_s
//       but K is never materialized; only its four weighted sums).
// Finally mirror the upper triangle to the lower (Omega is symmetric).

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
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

// Squared L2 row-norm of Z, one entry per row of the jagged stack.
__global__ void compute_znorms_kernel(const double* __restrict__ Z,
                                      int64_t total_rows, int64_t K,
                                      double* __restrict__ znorms) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total_rows) return;
  double s = 0.0;
  for (int64_t k = 0; k < K; ++k) {
    const double z = Z[i * K + k];
    s += z * z;
  }
  znorms[i] = s;
}

// Per-month sum of r_t, precomputed once.
__global__ void compute_sum_r_kernel(const double* __restrict__ r,
                                     const int64_t* __restrict__ offsets,
                                     int64_t T, double* __restrict__ sum_r) {
  int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (t >= T) return;
  const int64_t b = offsets[t], e = offsets[t + 1];
  double s = 0.0;
  for (int64_t i = b; i < e; ++i) s += r[i];
  sum_r[t] = s;
}

// Reflect the strict upper triangle into the strict lower triangle.
__global__ void mirror_upper_to_lower_kernel(double* Omega, int64_t T) {
  int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= T * T) return;
  const int64_t i = idx / T, j = idx % T;
  if (i > j) Omega[idx] = Omega[j * T + i];
}

// One block per pair (t, s). blockDim.x threads share the (i, j) work.
// Each thread accumulates 4 partial sums; then a tree-reduce inside shared
// memory yields the final 4 scalars. The closed-form combination is computed
// by thread 0 and written to *out (one device fp64).
template <int NT>
__global__ void rbf_fused_pair_kernel(const double* __restrict__ M,
                                      const double* __restrict__ z1n,
                                      const double* __restrict__ z2n,
                                      const double* __restrict__ r1,
                                      const double* __restrict__ r2,
                                      int64_t N1, int64_t N2,
                                      double inv_2c2,
                                      double sum_r1, double sum_r2,
                                      double* __restrict__ out) {
  static_assert(NT > 0 && (NT & (NT - 1)) == 0, "NT must be a power of two");
  __shared__ double sh_k[NT];
  __shared__ double sh_k1[NT];
  __shared__ double sh_1k[NT];
  __shared__ double sh_ab[NT];

  const int tid = threadIdx.x;
  const int64_t total = N1 * N2;

  double s_k = 0.0, s_k1 = 0.0, s_1k = 0.0, s_ab = 0.0;
  for (int64_t idx = tid; idx < total; idx += NT) {
    const int64_t i = idx / N2;
    const int64_t j = idx - i * N2;
    const double sq = z1n[i] + z2n[j] - 2.0 * M[idx];
    const double k  = exp(-sq * inv_2c2);
    const double ri = r1[i];
    const double rj = r2[j];
    s_k  += k;
    s_k1 += ri * k;
    s_1k += k  * rj;
    s_ab += ri * k * rj;
  }
  sh_k[tid] = s_k;   sh_k1[tid] = s_k1;
  sh_1k[tid] = s_1k; sh_ab[tid] = s_ab;
  __syncthreads();

  for (int stride = NT / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_k[tid]  += sh_k[tid  + stride];
      sh_k1[tid] += sh_k1[tid + stride];
      sh_1k[tid] += sh_1k[tid + stride];
      sh_ab[tid] += sh_ab[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const double S_K  = sh_k[0],  S_K1 = sh_k1[0];
    const double S_1K = sh_1k[0], S_ab = sh_ab[0];
    const double n1 = static_cast<double>(N1);
    const double n2 = static_cast<double>(N2);
    *out = S_ab
         - S_K1 * sum_r2 / n2
         - sum_r1 * S_1K / n1
         + S_K  * sum_r1 * sum_r2 / (n1 * n2);
  }
}

static void mkdir_p(const std::string& dir) {
  const std::string cmd = "mkdir -p '" + dir + "'";
  if (std::system(cmd.c_str()) != 0)
    throw std::runtime_error("mkdir failed: " + dir);
}

int main(int argc, char** argv) {
  const std::string prep_dir = (argc > 1) ? argv[1] : "data/prep";
  const std::string out_dir  = (argc > 2) ? argv[2] : "data/out";
  const double c = (argc > 3) ? std::atof(argv[3]) : 0.0058780160722749115;
  const std::string out_path = out_dir + "/omega_rbf_" +
                               (argc > 3 ? std::string(argv[3]) :
                                "0.0058780160722749115") +
                               ".f64.bin";

  ksdf::PrepData p = ksdf::load_prep(prep_dir);
  const int64_t T = p.T, K = p.K, R = p.total_rows;
  std::printf("[omega-rbf] T=%lld K=%lld total_rows=%lld c=%.17g\n",
              (long long)T, (long long)K, (long long)R, c);

  int64_t max_N = 0;
  for (int64_t t = 0; t < T; ++t) max_N = std::max(max_N, p.n_t(t));
  std::printf("[omega-rbf] max N_t = %lld (M buffer = %.1f MB)\n",
              (long long)max_N,
              (double)(max_N * max_N * sizeof(double)) / (1024.0 * 1024.0));

  double  *dZ = nullptr, *dR = nullptr, *dZN = nullptr, *dSR = nullptr;
  double  *dM = nullptr, *dOmega = nullptr;
  int64_t *dOff = nullptr;
  CUDA_CHECK(cudaMalloc(&dZ,    R * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dR,    R     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOff,  (T + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&dZN,   R     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dSR,   T     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dM,    max_N * max_N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOmega, T * T * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(dZ,   p.Z,       R * K * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dR,   p.r,       R     * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dOff, p.offsets, (T + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOmega, 0, T * T * sizeof(double)));

  compute_znorms_kernel<<<static_cast<unsigned>((R + 255) / 256), 256>>>(dZ, R, K, dZN);
  compute_sum_r_kernel<<<static_cast<unsigned>((T + 255) / 256), 256>>>(dR, dOff, T, dSR);
  CUDA_CHECK(cudaGetLastError());

  std::vector<int64_t> off_h(T + 1);
  std::vector<double>  sum_r_h(T);
  CUDA_CHECK(cudaMemcpy(off_h.data(),   dOff, (T + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sum_r_h.data(), dSR,  T       * sizeof(double),  cudaMemcpyDeviceToHost));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const double inv_2c2 = 1.0 / (2.0 * c * c);
  const double alpha = 1.0, beta = 0.0;
  constexpr int NT = 256;

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  int64_t pairs_done = 0;
  const int64_t pairs_total = T * (T + 1) / 2;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t off_t = off_h[t];
    const int64_t n1    = off_h[t + 1] - off_t;
    for (int64_t s = t; s < T; ++s) {
      const int64_t off_s = off_h[s];
      const int64_t n2    = off_h[s + 1] - off_s;

      // M (row-major n1 x n2) = Z_t @ Z_s^T, computed in col-major view as
      // M_cm (n2 x n1) = (Z_s_row^T)^T @ (Z_t_row^T) per the row/col reasoning
      // documented in omega_linear.cu. See also notes in step-4 commit message.
      CUBLAS_CHECK(cublasDgemm(handle,
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               /*m=*/n2, /*n=*/n1, /*k=*/K,
                               &alpha,
                               /*A=*/dZ + off_s * K, /*lda=*/K,
                               /*B=*/dZ + off_t * K, /*ldb=*/K,
                               &beta,
                               /*C=*/dM, /*ldc=*/n2));

      rbf_fused_pair_kernel<NT><<<1, NT>>>(
          dM, dZN + off_t, dZN + off_s,
          dR  + off_t, dR  + off_s,
          n1, n2, inv_2c2,
          sum_r_h[t], sum_r_h[s],
          dOmega + t * T + s);
      ++pairs_done;
    }
    if ((t & 31) == 31 || t == T - 1) {
      std::fprintf(stderr, "\r[omega-rbf] t=%lld/%lld  pairs=%lld/%lld",
                   (long long)(t + 1), (long long)T,
                   (long long)pairs_done, (long long)pairs_total);
      std::fflush(stderr);
    }
  }
  std::fprintf(stderr, "\n");

  const unsigned mblocks = static_cast<unsigned>((T * T + 255) / 256);
  mirror_upper_to_lower_kernel<<<mblocks, 256>>>(dOmega, T);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  std::printf("[omega-rbf] pairs+mirror = %.2f s (%.1f kpair/s)\n",
              ms / 1000.0f, (float)pairs_total / ms);

  std::vector<double> h_omega(static_cast<size_t>(T) * T);
  CUDA_CHECK(cudaMemcpy(h_omega.data(), dOmega,
                       T * T * sizeof(double), cudaMemcpyDeviceToHost));

  mkdir_p(out_dir);
  std::ofstream fout(out_path, std::ios::binary);
  if (!fout) throw std::runtime_error("open(" + out_path + ") failed");
  fout.write(reinterpret_cast<const char*>(h_omega.data()),
             static_cast<std::streamsize>(h_omega.size() * sizeof(double)));
  if (!fout) throw std::runtime_error("write(" + out_path + ") failed");
  std::printf("[omega-rbf] wrote %s (%zu bytes)\n",
              out_path.c_str(), h_omega.size() * sizeof(double));

  cublasDestroy(handle);
  cudaFree(dOmega);
  cudaFree(dM);
  cudaFree(dSR);
  cudaFree(dZN);
  cudaFree(dOff);
  cudaFree(dR);
  cudaFree(dZ);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  ksdf::free_prep(p);
  return 0;
}
