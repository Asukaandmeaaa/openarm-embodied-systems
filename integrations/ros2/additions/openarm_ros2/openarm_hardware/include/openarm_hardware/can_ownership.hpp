#pragma once

// Cooperative, process-wide CAN writer exclusion. This class never opens CAN.
// Never unlink an ownership file: doing so would permit locking two inodes.
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace openarm_hardware {

class CanOwnership {
 public:
  explicit CanOwnership(const std::string& interface, const std::string& backend
#ifdef OPENARM_CAN_OWNERSHIP_TESTING
                        , const std::string& root
#endif
                        ) {
#ifndef OPENARM_CAN_OWNERSHIP_TESTING
    const std::string root = "/run/lock/openarm-can";
#endif
    if (interface.empty() || interface.size() > 15 || interface == "." ||
        interface == ".." || interface.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos) {
      throw std::runtime_error("Invalid CAN interface name for ownership lock");
    }
    if (::mkdir(root.c_str(), 0755) != 0 && errno != EEXIST) {
      throw std::runtime_error("Cannot provision CAN ownership directory " + root + ": " + std::strerror(errno));
    }
    int dir = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (dir < 0) throw std::runtime_error("Cannot open CAN ownership directory " + root);
    struct stat ds {};
    if (::fstat(dir, &ds) != 0 || !S_ISDIR(ds.st_mode) || (ds.st_mode & 0022) ||
        (ds.st_uid != ::geteuid() && ds.st_uid != 0)) {
      ::close(dir);
      throw std::runtime_error("Unsafe CAN ownership directory permissions/owner: " + root);
    }
    path_ = root + "/" + interface + ".lock";
    fd_ = ::openat(dir, (interface + ".lock").c_str(),
                   O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0644);
    ::close(dir);
    if (fd_ < 0) throw std::runtime_error("Cannot open CAN ownership file " + path_ + ": " + std::strerror(errno));
    try {
      struct stat fs {};
      if (::fstat(fd_, &fs) != 0 || !S_ISREG(fs.st_mode) || fs.st_nlink != 1 ||
          (fs.st_mode & 0022) || (fs.st_uid != ::geteuid() && fs.st_uid != 0)) {
        throw std::runtime_error("Unsafe CAN ownership file: " + path_);
      }
      if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        char owner[4097] {};
        const ssize_t size = ::pread(fd_, owner, sizeof(owner) - 1, 0);
        throw std::runtime_error("CAN interface " + interface + " is already owned; refusing hardware initialization. Owner: " +
                                 (size > 0 ? std::string(owner, size) : "unavailable"));
      }
      // Legacy writers predate this lock protocol. Existing receiver entries
      // can be writers, so conservatively refuse BEFORE SDK/socket creation.
      // Test builds inspect only their explicitly supplied private directory.
#ifdef OPENARM_CAN_OWNERSHIP_TESTING
      reject_existing_receivers(interface, root, false);
#else
      reject_existing_receivers(interface, "/proc/net/can", true);
#endif
      const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
      const std::string record = "{\"schema\":1,\"pid\":" + std::to_string(::getpid()) +
        ",\"uid\":" + std::to_string(::geteuid()) + ",\"backend\":" + quote(backend) +
        ",\"interface\":" + quote(interface) + ",\"acquired_unix_ns\":" + std::to_string(now) + "}\n";
      if (::ftruncate(fd_, 0) != 0 || ::pwrite(fd_, record.data(), record.size(), 0) != static_cast<ssize_t>(record.size()) ||
          ::fsync(fd_) != 0) {
        throw std::runtime_error("Cannot write CAN ownership metadata: " + path_);
      }
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }
  ~CanOwnership() { if (fd_ >= 0) ::close(fd_); }
  CanOwnership(const CanOwnership&) = delete;
  CanOwnership& operator=(const CanOwnership&) = delete;
  const std::string& path() const { return path_; }

 private:
  static void reject_existing_receivers(const std::string& interface,
                                       const std::string& root, bool require_files) {
    bool saw_file = false;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
      if (entry.path().filename().string().rfind("rcvlist_", 0) != 0) continue;
      saw_file = true;
      std::ifstream stream(entry.path());
      if (!stream.is_open()) throw std::runtime_error("Cannot inspect legacy CAN receivers: " + entry.path().string());
      std::string line;
      while (std::getline(stream, line)) {
        std::istringstream row(line);
        std::string device;
        row >> device;
        if (device == interface || device == "(any)" || device == "any") {
          throw std::runtime_error("Existing CAN receiver may be an unpatched writer on " + interface +
            "; refusing hardware initialization. Stop legacy writers/recorders before startup. Source: " +
            entry.path().string() + " " + line);
        }
      }
      if (stream.bad()) throw std::runtime_error("Failed reading legacy CAN receiver registry");
    }
    if (require_files && !saw_file) {
      throw std::runtime_error("Cannot verify legacy CAN writers: receiver registry unavailable");
    }
  }
  static std::string quote(const std::string& value) {
    std::string out = "\"";
    for (const unsigned char c : value) {
      if (c == '"' || c == '\\') { out += '\\'; out += c; }
      else if (c >= 32 && c < 127) out += c;
      else out += '?';
    }
    return out + '"';
  }
  int fd_{-1};
  std::string path_;
};
}  // namespace openarm_hardware
