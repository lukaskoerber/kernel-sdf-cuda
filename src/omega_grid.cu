// omega_grid: CK-PCA Omega over a grid of c values, for the RBF and poly2
// kernels, on the A100.
//
// All kernels share the reference's double-centered quadratic form
//   K_tilde[i,j] = K[i,j] - rowmean_i - colmean_j + allmean
//   Omega[t,s]   = r_t' K_tilde r_s
// which, expanded, needs only four weighted sums per pair (S_K, S_K1, S_1K,
// S_ab) -- K is never materialized. Only the per-element kernel value differs:
//   rbf:   K = exp(-(||x||^2 + ||y||^2 - 2 M)/(2 c^2)),  M = x . y
//   poly2: K = (M + c)^2                                 (degree 2, coef0 = c)
//
// This reuses the per-month batched-GEMM structure of omega_rbf_opt.cu: one big
// tensor-core Dgemm per month t (M_t = Z_t @ Z_[t:]') feeds one reduction per
// partner s. Crucially M_t is computed ONCE per month and reused across the
// whole c-grid -- 614 GEMMs total regardless of grid size; only the (cheap)
// reductions repeat per c. Two streams + double-buffered M_t overlap GEMM(t+1)
// with the reductions of t.

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

enum KernelKind { KERNEL_RBF = 0, KERNEL_POLY2 = 1 };

// Squared L2 row-norm of Z, one entry per row (used by the RBF kernel only).
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

// Reduction for a whole month t, fanned out over all partners s >= t, for ONE
// value of the kernel hyperparameter. Identical structure for both kernels;
// the KIND template selects the per-element kernel value at compile time.
//   grid = (T - t, BPP), block = NT threads
//   M_t[i*C + c] = Z_t[i] . Z[off_t + c]      (C = R - off_t)
//   param = c (poly2 coef0) or inv_2c2 = 1/(2 c^2) (rbf)
template <int KIND, int NT>
__global__ void reduce_month_kernel(const double* __restrict__ Mt,
                                    const double* __restrict__ ZN,
                                    const double* __restrict__ Rv,
                                    const int64_t* __restrict__ offsets,
                                    const double* __restrict__ sum_r,
                                    int64_t t, int64_t T, int64_t off_t,
                                    int64_t C, double param,
                                    double* __restrict__ Omega) {
  const int64_t s = t + blockIdx.x;
  if (s >= T) return;

  const int64_t off_s = offsets[s];
  const int64_t n1    = offsets[t + 1] - off_t;
  const int64_t n2    = offsets[s + 1] - off_s;
  const int64_t cbase = off_s - off_t;
  const double  Rt = sum_r[t], Rs = sum_r[s];

  const int tid = threadIdx.x;

  double s_k = 0.0, s_k1 = 0.0, s_1k = 0.0, s_ab = 0.0;
  for (int64_t i = blockIdx.y; i < n1; i += gridDim.y) {
    const double  ri   = Rv[off_t + i];
    const double* Mrow = Mt + i * C + cbase;
    const double  zni  = (KIND == KERNEL_RBF) ? ZN[off_t + i] : 0.0;
    for (int64_t j = tid; j < n2; j += NT) {
      const double m = Mrow[j];
      double k;
      if (KIND == KERNEL_RBF) {
        const double sq = zni + ZN[off_s + j] - 2.0 * m;
        k = exp(-sq * param);               // param = 1/(2 c^2)
      } else {                              // KERNEL_POLY2
        const double base = m + param;      // param = coef0 = c
        k = base * base;
      }
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
  if (argc < 5) {
    std::fprintf(stderr,
        "usage: %s <rbf|poly2> <prep_dir> <out_dir> <c1> [c2 c3 ...]\n",
        argv[0]);
    return 2;
  }
  const std::string kernel   = argv[1];
  const std::string prep_dir = argv[2];
  const std::string out_dir  = argv[3];

  int kind;
  if (kernel == "rbf")        kind = KERNEL_RBF;
  else if (kernel == "poly2") kind = KERNEL_POLY2;
  else { std::fprintf(stderr, "unknown kernel '%s'\n", kernel.c_str()); return 2; }

  // Keep the original argv strings for output filenames (Python controls the
  // exact precision/naming); parse to double for the computation.
  std::vector<std::string> c_strs;
  std::vector<double>       c_vals;
  for (int a = 4; a < argc; ++a) {
    c_strs.emplace_back(argv[a]);
    c_vals.push_back(std::atof(argv[a]));
  }
  const int G = static_cast<int>(c_vals.size());

  ksdf::PrepData p = ksdf::load_prep(prep_dir);
  const int64_t T = p.T, K = p.K, R = p.total_rows;
  std::printf("[omega-grid] kernel=%s T=%lld K=%lld total_rows=%lld grid=%d\n",
              kernel.c_str(), (long long)T, (long long)K, (long long)R, G);

  std::vector<int64_t> off_h(T + 1);
  std::memcpy(off_h.data(), p.offsets, (T + 1) * sizeof(int64_t));
  int64_t max_N = 0;
  for (int64_t t = 0; t < T; ++t) max_N = std::max(max_N, off_h[t + 1] - off_h[t]);
  const int64_t mbuf_elems = max_N * R;

  double  *dZ = nullptr, *dR = nullptr, *dZN = nullptr, *dSR = nullptr;
  int64_t *dOff = nullptr;
  double  *dM[2] = {nullptr, nullptr};
  CUDA_CHECK(cudaMalloc(&dZ,   R * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dR,   R     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOff, (T + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&dZN,  R     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dSR,  T     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dM[0], mbuf_elems * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dM[1], mbuf_elems * sizeof(double)));

  // One Omega accumulator per grid point (T*T*8 = ~3 MB each).
  std::vector<double*> dOmega(G, nullptr);
  for (int g = 0; g < G; ++g) {
    CUDA_CHECK(cudaMalloc(&dOmega[g], T * T * sizeof(double)));
    CUDA_CHECK(cudaMemset(dOmega[g], 0, T * T * sizeof(double)));
  }

  CUDA_CHECK(cudaMemcpy(dZ,   p.Z,       R * K * sizeof(double),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dR,   p.r,       R     * sizeof(double),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dOff, p.offsets, (T + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));

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

  // Per-grid-point reduction parameter: rbf wants 1/(2 c^2); poly2 wants c.
  std::vector<double> param(G);
  for (int g = 0; g < G; ++g)
    param[g] = (kind == KERNEL_RBF) ? 1.0 / (2.0 * c_vals[g] * c_vals[g])
                                    : c_vals[g];

  const double alpha = 1.0, beta = 0.0;
  constexpr int NT  = 256;
  constexpr int BPP = 16;

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  for (int64_t t = 0; t < T; ++t) {
    const int b         = static_cast<int>(t & 1);
    const int64_t off_t = off_h[t];
    const int64_t n1    = off_h[t + 1] - off_t;
    const int64_t Ccols = R - off_t;

    // M_t = Z_t @ Z_[t:]' -- computed once, reused for every grid point.
    CUBLAS_CHECK(cublasDgemm(handle[b],
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             /*m=*/Ccols, /*n=*/n1, /*k=*/K,
                             &alpha,
                             dZ + off_t * K, /*lda=*/K,
                             dZ + off_t * K, /*ldb=*/K,
                             &beta,
                             dM[b], /*ldc=*/Ccols));

    dim3 grid(static_cast<unsigned>(T - t), BPP);
    for (int g = 0; g < G; ++g) {
      if (kind == KERNEL_RBF)
        reduce_month_kernel<KERNEL_RBF, NT><<<grid, NT, 0, stream[b]>>>(
            dM[b], dZN, dR, dOff, dSR, t, T, off_t, Ccols, param[g], dOmega[g]);
      else
        reduce_month_kernel<KERNEL_POLY2, NT><<<grid, NT, 0, stream[b]>>>(
            dM[b], dZN, dR, dOff, dSR, t, T, off_t, Ccols, param[g], dOmega[g]);
    }

    if ((t & 63) == 63 || t == T - 1) {
      std::fprintf(stderr, "\r[omega-grid] t=%lld/%lld", (long long)(t + 1), (long long)T);
      std::fflush(stderr);
    }
  }
  std::fprintf(stderr, "\n");
  CUDA_CHECK(cudaGetLastError());

  const unsigned mblocks = static_cast<unsigned>((T * T + 255) / 256);
  for (int g = 0; g < G; ++g)
    mirror_upper_to_lower_kernel<<<mblocks, 256>>>(dOmega[g], T);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  const int64_t pairs_total = T * (T + 1) / 2;
  std::printf("[omega-grid] total = %.3f s for %d grid point(s) "
              "(%.3f s/point, %.1f kpair/s/point)\n",
              ms / 1000.0f, G, ms / 1000.0f / G,
              (float)pairs_total / (ms / G));

  mkdir_p(out_dir);
  std::vector<double> h_omega(static_cast<size_t>(T) * T);
  for (int g = 0; g < G; ++g) {
    CUDA_CHECK(cudaMemcpy(h_omega.data(), dOmega[g],
                         T * T * sizeof(double), cudaMemcpyDeviceToHost));
    const std::string out_path =
        out_dir + "/omega_" + kernel + "_" + c_strs[g] + ".f64.bin";
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout) throw std::runtime_error("open(" + out_path + ") failed");
    fout.write(reinterpret_cast<const char*>(h_omega.data()),
               static_cast<std::streamsize>(h_omega.size() * sizeof(double)));
    if (!fout) throw std::runtime_error("write(" + out_path + ") failed");
    std::printf("[omega-grid] wrote %s\n", out_path.c_str());
  }

  for (int b = 0; b < 2; ++b) {
    cublasDestroy(handle[b]);
    cudaStreamDestroy(stream[b]);
    cudaFree(dM[b]);
  }
  for (int g = 0; g < G; ++g) cudaFree(dOmega[g]);
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
