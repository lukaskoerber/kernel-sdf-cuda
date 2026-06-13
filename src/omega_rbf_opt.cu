// omega_rbf_opt: optimized CK-PCA Omega for the RBF kernel on the A100.
//
// The baseline (omega_rbf.cu) processes the T(T+1)/2 month-pairs one at a
// time: a tiny per-pair Dgemm followed by a single-block reduction. On an
// A100 that leaves 107/108 SMs idle and pays ~377k kernel launches.
//
// This version restructures the work *per month t* instead of per pair:
//
//   1. ONE large Dgemm per t:  M_t = Z_t @ Z_[t:]'  (n_t x C, C = R - off_t)
//      i.e. Z_t against the entire tail of the stack (all partners s >= t at
//      once). 614 big tensor-core GEMMs replace 188k tiny ones.
//
//   2. ONE reduction kernel per t spanning every partner s >= t. The grid is
//      (T - t) x BLOCKS_PER_PAIR: blockIdx.x selects the partner s, blockIdx.y
//      splits that pair's (i, j) grid across SMs. Each block streams its slice
//      of M_t, forms K[i,j] = exp(-(zn_i + zn_j - 2 M)/2c^2) on the fly, and
//      accumulates the four weighted sums S_K, S_K1, S_1K, S_ab. Because
//      Omega[t,s] is *linear* in those four sums, each block folds its partials
//      into one partial Omega and atomicAdds it to Omega[t,s] -- one atomic per
//      block, no second pass, K never materialized.
//
//   Two streams + double-buffered M_t overlap GEMM(t+1) with reduce(t).
//
// Algebra is identical to the baseline; only the parallelization differs.
// Finally mirror the upper triangle into the lower (Omega is symmetric).

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

// Reduction for a whole month t, fanned out over all partners s >= t.
//   grid  = (T - t, BPP), block = NT threads
//   M_t   : (n1 x C) with M_t[i*C + c] = Z_t[i] . Z[off_t + c]   (C = R - off_t)
//   blockIdx.x -> partner s = t + blockIdx.x; blockIdx.y splits the (i,j) grid.
template <int NT>
__global__ void rbf_reduce_month_kernel(const double* __restrict__ Mt,
                                        const double* __restrict__ ZN,
                                        const double* __restrict__ Rv,
                                        const int64_t* __restrict__ offsets,
                                        const double* __restrict__ sum_r,
                                        int64_t t, int64_t T, int64_t off_t,
                                        int64_t C, double inv_2c2,
                                        double* __restrict__ Omega) {
  const int64_t s = t + blockIdx.x;
  if (s >= T) return;

  const int64_t off_s = offsets[s];
  const int64_t n1    = offsets[t + 1] - off_t;
  const int64_t n2    = offsets[s + 1] - off_s;
  const int64_t cbase = off_s - off_t;   // column offset of s-block within M_t
  const int64_t total = n1 * n2;
  const double  Rt = sum_r[t], Rs = sum_r[s];

  const int tid = threadIdx.x;
  (void)total;

  // blockIdx.y strides over rows i (no per-element division); threads stride
  // over columns j. Within a warp consecutive j keep Mrow[j]/ZN/Rv coalesced.
  double s_k = 0.0, s_k1 = 0.0, s_1k = 0.0, s_ab = 0.0;
  for (int64_t i = blockIdx.y; i < n1; i += gridDim.y) {
    const double  zni  = ZN[off_t + i];
    const double  ri   = Rv[off_t + i];
    const double* Mrow = Mt + i * C + cbase;
    for (int64_t j = tid; j < n2; j += NT) {
      const double sq = zni + ZN[off_s + j] - 2.0 * Mrow[j];
      const double k  = exp(-sq * inv_2c2);
      const double rj = Rv[off_s + j];
      s_k  += k;
      s_k1 += ri * k;
      s_1k += k  * rj;
      s_ab += ri * k * rj;
    }
  }

  __shared__ double sh_k[NT], sh_k1[NT], sh_1k[NT], sh_ab[NT];
  sh_k[tid] = s_k;   sh_k1[tid] = s_k1;
  sh_1k[tid] = s_1k; sh_ab[tid] = s_ab;
  __syncthreads();
  for (int st = NT / 2; st > 0; st >>= 1) {
    if (tid < st) {
      sh_k[tid]  += sh_k[tid  + st];
      sh_k1[tid] += sh_k1[tid + st];
      sh_1k[tid] += sh_1k[tid + st];
      sh_ab[tid] += sh_ab[tid + st];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const double n1d = static_cast<double>(n1);
    const double n2d = static_cast<double>(n2);
    const double partial = sh_ab[0]
                         - sh_k1[0] * Rs / n2d
                         - Rt * sh_1k[0] / n1d
                         + sh_k[0] * Rt * Rs / (n1d * n2d);
    atomicAdd(&Omega[t * T + s], partial);
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
  const std::string c_str = (argc > 3) ? std::string(argv[3])
                                       : "0.0058780160722749115";
  const std::string out_path = out_dir + "/omega_rbf_opt_" + c_str + ".f64.bin";

  ksdf::PrepData p = ksdf::load_prep(prep_dir);
  const int64_t T = p.T, K = p.K, R = p.total_rows;
  std::printf("[omega-rbf-opt] T=%lld K=%lld total_rows=%lld c=%.17g\n",
              (long long)T, (long long)K, (long long)R, c);

  std::vector<int64_t> off_h(T + 1);
  std::memcpy(off_h.data(), p.offsets, (T + 1) * sizeof(int64_t));
  int64_t max_N = 0;
  for (int64_t t = 0; t < T; ++t) max_N = std::max(max_N, off_h[t + 1] - off_h[t]);

  // M_t buffer is (n_t x C); the worst case is t=0 (C = R, n_t up to max_N).
  const int64_t mbuf_elems = max_N * R;
  std::printf("[omega-rbf-opt] max N_t=%lld  M_t buffer=%.2f GB x2 (double-buffered)\n",
              (long long)max_N,
              (double)(mbuf_elems * sizeof(double)) / (1024.0 * 1024.0 * 1024.0));

  double  *dZ = nullptr, *dR = nullptr, *dZN = nullptr, *dSR = nullptr;
  double  *dOmega = nullptr;
  int64_t *dOff = nullptr;
  double  *dM[2] = {nullptr, nullptr};
  CUDA_CHECK(cudaMalloc(&dZ,    R * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dR,    R     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOff,  (T + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&dZN,   R     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dSR,   T     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOmega, T * T * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dM[0], mbuf_elems * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dM[1], mbuf_elems * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(dZ,   p.Z,       R * K * sizeof(double),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dR,   p.r,       R     * sizeof(double),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dOff, p.offsets, (T + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOmega, 0, T * T * sizeof(double)));

  compute_znorms_kernel<<<static_cast<unsigned>((R + 255) / 256), 256>>>(dZ, R, K, dZN);
  compute_sum_r_kernel<<<static_cast<unsigned>((T + 255) / 256), 256>>>(dR, dOff, T, dSR);
  CUDA_CHECK(cudaGetLastError());

  cudaStream_t stream[2];
  cublasHandle_t handle[2];
  for (int b = 0; b < 2; ++b) {
    CUDA_CHECK(cudaStreamCreate(&stream[b]));
    CUBLAS_CHECK(cublasCreate(&handle[b]));
    CUBLAS_CHECK(cublasSetStream(handle[b], stream[b]));
  }

  const double inv_2c2 = 1.0 / (2.0 * c * c);
  const double alpha = 1.0, beta = 0.0;
  constexpr int NT  = 256;
  constexpr int BPP = 16;   // blocks per pair (splits one pair across SMs)

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  for (int64_t t = 0; t < T; ++t) {
    const int b        = static_cast<int>(t & 1);
    const int64_t off_t = off_h[t];
    const int64_t n1    = off_h[t + 1] - off_t;
    const int64_t Ccols = R - off_t;          // partners s >= t span the tail

    // M_t = Z_t @ Z_[t:]'  ->  dM[b][i*Ccols + c] = Z_t[i] . Z[off_t + c].
    // Column-major (Ccols x n1) with ldc=Ccols matches that row-major view.
    CUBLAS_CHECK(cublasDgemm(handle[b],
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             /*m=*/Ccols, /*n=*/n1, /*k=*/K,
                             &alpha,
                             /*A=*/dZ + off_t * K, /*lda=*/K,
                             /*B=*/dZ + off_t * K, /*ldb=*/K,
                             &beta,
                             /*C=*/dM[b], /*ldc=*/Ccols));

    dim3 grid(static_cast<unsigned>(T - t), BPP);
    rbf_reduce_month_kernel<NT><<<grid, NT, 0, stream[b]>>>(
        dM[b], dZN, dR, dOff, dSR,
        t, T, off_t, Ccols, inv_2c2, dOmega);

    if ((t & 63) == 63 || t == T - 1) {
      std::fprintf(stderr, "\r[omega-rbf-opt] t=%lld/%lld", (long long)(t + 1), (long long)T);
      std::fflush(stderr);
    }
  }
  std::fprintf(stderr, "\n");
  CUDA_CHECK(cudaGetLastError());

  const unsigned mblocks = static_cast<unsigned>((T * T + 255) / 256);
  mirror_upper_to_lower_kernel<<<mblocks, 256>>>(dOmega, T);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  const int64_t pairs_total = T * (T + 1) / 2;
  std::printf("[omega-rbf-opt] gemm+reduce+mirror = %.3f s (%.1f kpair/s)\n",
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
  std::printf("[omega-rbf-opt] wrote %s (%zu bytes)\n",
              out_path.c_str(), h_omega.size() * sizeof(double));

  for (int b = 0; b < 2; ++b) {
    cublasDestroy(handle[b]);
    cudaStreamDestroy(stream[b]);
    cudaFree(dM[b]);
  }
  cudaFree(dOmega);
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
