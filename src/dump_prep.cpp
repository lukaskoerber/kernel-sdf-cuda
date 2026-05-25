// dump_prep: load data/prep/ and print summary + a single month's head rows.
// Usage: dump_prep [dir=data/prep] [t=0]

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "prep_loader.hpp"

int main(int argc, char** argv) {
  const std::string dir = (argc > 1) ? argv[1] : "data/prep";
  const int64_t t_show  = (argc > 2) ? std::stoll(argv[2]) : 0;

  ksdf::PrepData p;
  try {
    p = ksdf::load_prep(dir);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "load_prep failed: %s\n", e.what());
    return 1;
  }

  if (t_show < 0 || t_show >= p.T) {
    std::fprintf(stderr, "t=%lld out of range [0, %lld)\n",
                 static_cast<long long>(t_show),
                 static_cast<long long>(p.T));
    ksdf::free_prep(p);
    return 1;
  }

  std::printf("prep dir   : %s\n", dir.c_str());
  std::printf("T          : %lld\n", (long long)p.T);
  std::printf("K          : %lld\n", (long long)p.K);
  std::printf("total_rows : %lld\n", (long long)p.total_rows);
  std::printf("dates      : %s .. %s\n",
              ksdf::format_date_d(p.dates[0]).c_str(),
              ksdf::format_date_d(p.dates[p.T - 1]).c_str());

  const int64_t n     = p.n_t(t_show);
  const double* Zt    = p.Z_row(t_show);
  const double* rt    = p.r_row(t_show);

  std::printf("\nt=%lld date=%s N_t=%lld offsets=[%lld, %lld)\n",
              (long long)t_show,
              ksdf::format_date_d(p.dates[t_show]).c_str(),
              (long long)n,
              (long long)p.offsets[t_show],
              (long long)p.offsets[t_show + 1]);

  std::printf("Z[t][0][0..5] =");
  for (int j = 0; j < 5 && j < p.K; ++j)
    std::printf(" % .9e", Zt[j]);
  std::printf("\n");

  std::printf("Z[t][N_t-1][0..5] =");
  const double* last_row = Zt + (n - 1) * p.K;
  for (int j = 0; j < 5 && j < p.K; ++j)
    std::printf(" % .9e", last_row[j]);
  std::printf("\n");

  std::printf("r[t][0..5] =");
  for (int j = 0; j < 5 && j < n; ++j)
    std::printf(" % .9e", rt[j]);
  std::printf("\n");

  ksdf::free_prep(p);
  return 0;
}
