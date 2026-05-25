#include "prep_loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>

namespace ksdf {

namespace {

PrepData::MMap mmap_ro(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("open(" + path + "): " + std::strerror(errno));
  struct stat st {};
  if (::fstat(fd, &st) < 0) {
    ::close(fd);
    throw std::runtime_error("fstat(" + path + "): " + std::strerror(errno));
  }
  if (st.st_size == 0) {
    ::close(fd);
    throw std::runtime_error("empty file: " + path);
  }
  void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                   MAP_PRIVATE, fd, 0);
  ::close(fd);  // mapping survives close on Linux/POSIX
  if (p == MAP_FAILED)
    throw std::runtime_error("mmap(" + path + "): " + std::strerror(errno));
  return PrepData::MMap{p, static_cast<size_t>(st.st_size)};
}

void unmap(PrepData::MMap& m) {
  if (m.base) {
    ::munmap(m.base, m.size);
    m.base = nullptr;
    m.size = 0;
  }
}

}  // namespace

PrepData load_prep(const std::string& dir) {
  PrepData p;
  p.z_map   = mmap_ro(dir + "/Z.f64.bin");
  p.r_map   = mmap_ro(dir + "/r.f64.bin");
  p.off_map = mmap_ro(dir + "/offsets.i64.bin");
  p.dt_map  = mmap_ro(dir + "/dates.i64.bin");

  if (p.r_map.size % sizeof(double) != 0)
    throw std::runtime_error("r.f64.bin size not a multiple of 8");
  if (p.off_map.size % sizeof(int64_t) != 0)
    throw std::runtime_error("offsets.i64.bin size not a multiple of 8");
  if (p.dt_map.size % sizeof(int64_t) != 0)
    throw std::runtime_error("dates.i64.bin size not a multiple of 8");
  if (p.z_map.size % sizeof(double) != 0)
    throw std::runtime_error("Z.f64.bin size not a multiple of 8");

  p.total_rows = static_cast<int64_t>(p.r_map.size / sizeof(double));
  const int64_t off_count = static_cast<int64_t>(p.off_map.size / sizeof(int64_t));
  const int64_t dt_count  = static_cast<int64_t>(p.dt_map.size  / sizeof(int64_t));
  p.T = off_count - 1;
  if (dt_count != p.T)
    throw std::runtime_error("inconsistent T: offsets implies " +
                             std::to_string(p.T) + ", dates has " +
                             std::to_string(dt_count));

  const int64_t z_doubles = static_cast<int64_t>(p.z_map.size / sizeof(double));
  if (p.total_rows == 0 || z_doubles % p.total_rows != 0)
    throw std::runtime_error("Z.f64.bin not divisible by total_rows");
  p.K = z_doubles / p.total_rows;

  p.Z       = static_cast<const double*>(p.z_map.base);
  p.r       = static_cast<const double*>(p.r_map.base);
  p.offsets = static_cast<const int64_t*>(p.off_map.base);
  p.dates   = static_cast<const int64_t*>(p.dt_map.base);

  if (p.offsets[0] != 0)
    throw std::runtime_error("offsets[0] != 0");
  if (p.offsets[p.T] != p.total_rows)
    throw std::runtime_error("offsets[T] (" + std::to_string(p.offsets[p.T]) +
                             ") != total_rows (" +
                             std::to_string(p.total_rows) + ")");
  for (int64_t t = 0; t < p.T; ++t) {
    if (p.offsets[t + 1] < p.offsets[t])
      throw std::runtime_error("offsets not monotonic at t=" + std::to_string(t));
  }
  for (int64_t t = 1; t < p.T; ++t) {
    if (p.dates[t] <= p.dates[t - 1])
      throw std::runtime_error("dates not strictly increasing at t=" +
                               std::to_string(t));
  }

  return p;
}

void free_prep(PrepData& p) {
  unmap(p.z_map);
  unmap(p.r_map);
  unmap(p.off_map);
  unmap(p.dt_map);
  p.Z = nullptr; p.r = nullptr; p.offsets = nullptr; p.dates = nullptr;
  p.T = p.K = p.total_rows = 0;
}

std::string format_date_d(int64_t days_since_epoch) {
  time_t secs = static_cast<time_t>(days_since_epoch) * 86400;
  struct tm tm {};
  gmtime_r(&secs, &tm);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tm.tm_year + 1900,
                tm.tm_mon + 1, tm.tm_mday);
  return buf;
}

}  // namespace ksdf
