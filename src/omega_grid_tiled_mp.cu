// omega_grid_tiled_mp: MIXED-PRECISION variant of omega_grid_tiled for the daily
// scale. Identical tiling/structure, but the expensive per-pair work is done in
// low precision while everything cancellation-prone stays fp64:
//
//   * GEMM (M = Z_grp @ Z_t')  -> TF32 tensor cores (CUBLAS_COMPUTE_32F_FAST_TF32),
//     fp32 inputs/outputs. This is the ~10^18-op bulk and the main speedup.
//   * znorms + squared-distance + exp -> fp32 (expf).
//   * returns, the double-centered 4-sum reduction, the accumulators, and the
//     T*T Omega output -> fp64 (centering is cancellation-prone).
//
// The output binary format is identical to omega_grid_tiled (row-major fp64
// T*T), so it diffs directly against the fp64 oracle. The fp64 engine remains
// the default/reference; this is an additive, parallel path.
//
// Env overrides (same as the fp64 engine): KSDF_ROW_BUDGET, KSDF_OMEGA_BATCH.

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

__global__ void f64_to_f32_kernel(const double* __restrict__ in,
                                  float* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = static_cast<float>(in[i]);
}

// znorms in fp32, from the same fp32 Z used in the TF32 GEMM (so the diagonal
// self-distance is consistent with the rounded inner products).
__global__ void compute_znorms_f32_kernel(const float* __restrict__ Z,
                                          int64_t total_rows, int64_t K,
                                          float* __restrict__ znorms) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total_rows) return;
  float s = 0.0f;
  for (int64_t k = 0; k < K; ++k) { const float z = Z[i * K + k]; s += z * z; }
  znorms[i] = s;
}

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

__global__ void mirror_upper_to_lower_kernel(double* Omega, int64_t T) {
  int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= T * T) return;
  const int64_t i = idx / T, j = idx % T;
  if (i > j) Omega[idx] = Omega[j * T + i];
}

// Reduction over one partner day-group tile. M / znorms are fp32; returns,
// accumulation, centering, and Omega are fp64.
template <int KIND, int NT>
__global__ void reduce_tiled_mp_kernel(const float* __restrict__ Mt,
                                       const float* __restrict__ ZN,
                                       const double* __restrict__ Rv,
                                       const int64_t* __restrict__ offsets,
                                       const double* __restrict__ sum_r,
                                       int64_t t, int64_t s0, int64_t off_t,
                                       int64_t tile_cols, float param,
                                       int64_t T, double* __restrict__ Omega) {
  const int64_t s        = s0 + blockIdx.x;
  const int64_t off_t2   = offsets[t];
  const int64_t off_s    = offsets[s];
  const int64_t off_grp  = offsets[s0];
  const int64_t n1       = offsets[t + 1] - off_t2;
  const int64_t n2       = offsets[s + 1] - off_s;
  const int64_t cbase    = off_s - off_grp;      // column offset within the tile
  const double  Rt = sum_r[t], Rs = sum_r[s];
  const int     tid = threadIdx.x;

  double s_k = 0.0, s_k1 = 0.0, s_1k = 0.0, s_ab = 0.0;
  for (int64_t i = blockIdx.y; i < n1; i += gridDim.y) {
    const double ri    = Rv[off_t + i];
    const float* Mrow  = Mt + i * tile_cols + cbase;
    const float  zni   = (KIND == KERNEL_RBF) ? ZN[off_t + i] : 0.0f;
    for (int64_t j = tid; j < n2; j += NT) {
      const float m = Mrow[j];
      float k;
      if (KIND == KERNEL_RBF) {
        const float sq = zni + ZN[off_s + j] - 2.0f * m;
        k = expf(-sq * param);
      } else {
        const float base = m + param;
        k = base * base;
      }
      const double kd = static_cast<double>(k);
      const double rj = Rv[off_s + j];
      s_k  += kd;
      s_k1 += ri * kd;
      s_1k += kd * rj;
      s_ab += ri * kd * rj;
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
    const double n1d = static_cast<double>(n1), n2d = static_cast<double>(n2);
    const double partial = sh_ab[0] - sh_k1[0] * Rs / n2d
                         - Rt * sh_1k[0] / n1d + sh_k[0] * Rt * Rs / (n1d * n2d);
    atomicAdd(&Omega[t * T + s], partial);
  }
}

static void mkdir_p(const std::string& dir) {
  const std::string cmd = "mkdir -p '" + dir + "'";
  if (std::system(cmd.c_str()) != 0) throw std::runtime_error("mkdir failed: " + dir);
}

static int64_t env_i64(const char* name, int64_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  return std::atoll(v);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
        "usage: %s <rbf|poly2> <prep_dir> <out_dir> <c1> [c2 ...]\n", argv[0]);
    return 2;
  }
  const std::string kernel = argv[1], prep_dir = argv[2], out_dir = argv[3];
  int kind;
  if (kernel == "rbf") kind = KERNEL_RBF;
  else if (kernel == "poly2") kind = KERNEL_POLY2;
  else { std::fprintf(stderr, "unknown kernel '%s'\n", kernel.c_str()); return 2; }

  std::vector<std::string> c_strs;
  std::vector<double>       c_vals;
  for (int a = 4; a < argc; ++a) { c_strs.emplace_back(argv[a]); c_vals.push_back(std::atof(argv[a])); }
  const int G = static_cast<int>(c_vals.size());

  ksdf::PrepData p = ksdf::load_prep(prep_dir);
  const int64_t T = p.T, K = p.K, R = p.total_rows;

  std::vector<int64_t> off_h(T + 1);
  std::memcpy(off_h.data(), p.offsets, (T + 1) * sizeof(int64_t));
  int64_t max_N = 0;
  for (int64_t t = 0; t < T; ++t) max_N = std::max(max_N, off_h[t + 1] - off_h[t]);

  // ---- auto-size the M-tile row budget and the c-batch from free memory ----
  // Steady-state device footprint: Z32(fp32) + r(f64) + znorms(f32) + sum_r + offsets.
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  const size_t fixed_b = static_cast<size_t>(R) * K * 4   // Z32
                       + static_cast<size_t>(R) * 8        // r
                       + static_cast<size_t>(R) * 4        // znorms (f32)
                       + static_cast<size_t>(T) * 8        // sum_r
                       + static_cast<size_t>(T + 1) * 8;   // offsets
  const size_t budget_b = static_cast<size_t>(free_b * 0.90) - fixed_b;
  // M tile is fp32 now (4 bytes). max_N*row_budget < 1.5e9 (int-index safe).
  int64_t row_budget = env_i64("KSDF_ROW_BUDGET", 0);
  if (row_budget <= 0) {
    int64_t by_mem = static_cast<int64_t>((budget_b * 0.40) / (4 * (double)max_N));
    int64_t by_idx = static_cast<int64_t>(1500000000.0 / (double)max_N);
    row_budget = std::min(by_mem, by_idx);
    row_budget = std::min(row_budget, R);
    row_budget = std::max(row_budget, max_N);   // at least one max day
  }
  const size_t mbuf_b = static_cast<size_t>(max_N) * row_budget * 4;
  const size_t omega_b = static_cast<size_t>(T) * T * 8;
  int omega_batch = static_cast<int>(env_i64("KSDF_OMEGA_BATCH", 0));
  if (omega_batch <= 0) {
    int64_t fit = static_cast<int64_t>((budget_b - mbuf_b) / omega_b);
    omega_batch = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(fit, G)));
  }
  omega_batch = std::min(omega_batch, G);

  std::printf("[omega-mp] kernel=%s T=%lld K=%lld rows=%lld grid=%d (TF32 GEMM)\n",
              kernel.c_str(), (long long)T, (long long)K, (long long)R, G);
  std::printf("[omega-mp] free=%.1f GB  row_budget=%lld (M tile %.2f GB fp32)  "
              "omega_batch=%d (%.2f GB/buf)\n",
              free_b / 1e9, (long long)row_budget, mbuf_b / 1e9, omega_batch,
              omega_b / 1e9);

  // Upload fp64 Z, convert to fp32, derive fp32 znorms, then free the fp64 copy.
  double  *dZ64=nullptr, *dR=nullptr, *dSR=nullptr;
  float   *dZ=nullptr, *dZN=nullptr, *dM=nullptr;
  int64_t *dOff=nullptr;
  CUDA_CHECK(cudaMalloc(&dZ64, R * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dZ,   R * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dR,   R * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dOff, (T + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&dZN,  R * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dSR,  T * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dM,   static_cast<size_t>(max_N) * row_budget * sizeof(float)));

  std::vector<double*> dOmega(omega_batch, nullptr);
  for (int g = 0; g < omega_batch; ++g)
    CUDA_CHECK(cudaMalloc(&dOmega[g], T * T * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(dZ64, p.Z,       R * K * sizeof(double),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dR,   p.r,       R * sizeof(double),        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dOff, p.offsets, (T + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  f64_to_f32_kernel<<<(unsigned)((R * K + 255) / 256), 256>>>(dZ64, dZ, R * K);
  compute_znorms_f32_kernel<<<(unsigned)((R + 255) / 256), 256>>>(dZ, R, K, dZN);
  compute_sum_r_kernel<<<(unsigned)((T + 255) / 256), 256>>>(dR, dOff, T, dSR);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(dZ64)); dZ64 = nullptr;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.0f, beta = 0.0f;
  constexpr int NT = 256, BPP = 16;

  std::vector<float> param(G);
  for (int g = 0; g < G; ++g)
    param[g] = (kind == KERNEL_RBF)
                   ? static_cast<float>(1.0 / (2.0 * c_vals[g] * c_vals[g]))
                   : static_cast<float>(c_vals[g]);

  mkdir_p(out_dir);
  std::vector<double> h_omega(static_cast<size_t>(T) * T);

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  for (int g0 = 0; g0 < G; g0 += omega_batch) {
    const int gb = std::min(omega_batch, G - g0);
    for (int g = 0; g < gb; ++g)
      CUDA_CHECK(cudaMemset(dOmega[g], 0, T * T * sizeof(double)));

    CUDA_CHECK(cudaEventRecord(e0));
    for (int64_t t = 0; t < T; ++t) {
      const int64_t off_t = off_h[t];
      const int64_t n1    = off_h[t + 1] - off_t;
      int64_t s0 = t;
      while (s0 < T) {
        const int64_t off_grp = off_h[s0];
        int64_t s_end = s0, grows = 0;
        while (s_end < T) {
          const int64_t nd = off_h[s_end + 1] - off_h[s_end];
          if (grows != 0 && grows + nd > row_budget) break;
          grows += nd; ++s_end;
        }
        // M tile (grows x n1) col-major = Z_grp @ Z_t' via TF32 tensor cores.
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  /*m=*/(int)grows, /*n=*/(int)n1, /*k=*/(int)K,
                                  &alpha,
                                  dZ + off_grp * K, CUDA_R_32F, (int)K,
                                  dZ + off_t * K,   CUDA_R_32F, (int)K,
                                  &beta,
                                  dM, CUDA_R_32F, /*ldc=*/(int)grows,
                                  CUBLAS_COMPUTE_32F_FAST_TF32,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        dim3 grid((unsigned)(s_end - s0), BPP);
        for (int g = 0; g < gb; ++g) {
          if (kind == KERNEL_RBF)
            reduce_tiled_mp_kernel<KERNEL_RBF, NT><<<grid, NT>>>(
                dM, dZN, dR, dOff, dSR, t, s0, off_t, grows, param[g0 + g], T, dOmega[g]);
          else
            reduce_tiled_mp_kernel<KERNEL_POLY2, NT><<<grid, NT>>>(
                dM, dZN, dR, dOff, dSR, t, s0, off_t, grows, param[g0 + g], T, dOmega[g]);
        }
        s0 = s_end;
      }
      if ((t & 1023) == 1023 || t == T - 1) {
        std::fprintf(stderr, "\r[omega-mp] batch %d-%d  t=%lld/%lld",
                     g0, g0 + gb - 1, (long long)(t + 1), (long long)T);
        std::fflush(stderr);
      }
    }
    std::fprintf(stderr, "\n");
    CUDA_CHECK(cudaGetLastError());

    const unsigned mblocks = (unsigned)((T * T + 255) / 256);
    for (int g = 0; g < gb; ++g)
      mirror_upper_to_lower_kernel<<<mblocks, 256>>>(dOmega[g], T);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    std::printf("[omega-mp] batch %d-%d: %.1f s (%.1f s/c-point)\n",
                g0, g0 + gb - 1, ms / 1000.0f, ms / 1000.0f / gb);

    for (int g = 0; g < gb; ++g) {
      CUDA_CHECK(cudaMemcpy(h_omega.data(), dOmega[g], T * T * sizeof(double),
                           cudaMemcpyDeviceToHost));
      const std::string out_path =
          out_dir + "/omega_" + kernel + "_" + c_strs[g0 + g] + ".f64.bin";
      std::ofstream fout(out_path, std::ios::binary);
      if (!fout) throw std::runtime_error("open(" + out_path + ") failed");
      fout.write(reinterpret_cast<const char*>(h_omega.data()),
                 (std::streamsize)(h_omega.size() * sizeof(double)));
      std::printf("[omega-mp] wrote %s\n", out_path.c_str());
    }
  }

  cublasDestroy(handle);
  for (int g = 0; g < omega_batch; ++g) cudaFree(dOmega[g]);
  cudaFree(dM); cudaFree(dSR); cudaFree(dZN); cudaFree(dOff); cudaFree(dR); cudaFree(dZ);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  ksdf::free_prep(p);
  return 0;
}
