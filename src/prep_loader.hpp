#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace ksdf {

// Read-only view over the mmapped data/prep/ files produced by python/prep.py.
// Sizes are derived from file sizes and cross-validated for consistency.
struct PrepData {
  const double*  Z       = nullptr;  // [total_rows, K] row-major fp64
  const double*  r       = nullptr;  // [total_rows]    fp64 (may contain NaN)
  const int64_t* offsets = nullptr;  // [T+1] indptr; month t in rows [off[t], off[t+1])
  const int64_t* dates   = nullptr;  // [T] datetime64[D] (days since 1970-01-01)

  int64_t T          = 0;
  int64_t K          = 0;
  int64_t total_rows = 0;

  struct MMap { void* base = nullptr; size_t size = 0; };
  MMap z_map, r_map, off_map, dt_map;

  const double*  Z_row(int64_t t) const { return Z + offsets[t] * K; }
  const double*  r_row(int64_t t) const { return r + offsets[t]; }
  int64_t        n_t  (int64_t t) const { return offsets[t + 1] - offsets[t]; }
};

PrepData load_prep(const std::string& dir);
void     free_prep(PrepData& p);

// "%Y-%m-%d" for datetime64[D] (days since 1970-01-01).
std::string format_date_d(int64_t days_since_epoch);

}  // namespace ksdf
