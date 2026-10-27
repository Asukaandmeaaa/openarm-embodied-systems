#pragma once

// Only used by the isolated vision-test startup. No CAN writes in this class.
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <linux/can.h>

namespace openarm_hardware {
struct RecoveryFeedback {
  std::array<double, 8> position{};
  std::array<double, 8> velocity{};
  std::array<unsigned, 8> status{};
  std::array<bool, 8> seen{};

  void accept(const canfd_frame& f) {
    if (f.len != 8 || f.can_id < 0x11 || f.can_id > 0x18) return;
    const unsigned i = f.can_id - 0x11;
    if ((f.data[0] & 15) != i + 1) return;
    const unsigned q = (unsigned(f.data[1]) << 8) | f.data[2];
    const unsigned dq = (unsigned(f.data[3]) << 4) | (f.data[4] >> 4);
    const double vmax = i < 2 ? 45. : (i < 4 ? 8. : 30.);
    position[i] = double(q) / 65535. * 25. - 12.5;
    velocity[i] = double(dq) / 4095. * (2. * vmax) - vmax;
    status[i] = f.data[0] >> 4;
    seen[i] = true;
  }

  void validate(unsigned required_status) const {
    for (unsigned i = 0; i < 8; ++i) {
      if (!seen[i]) throw std::runtime_error("Missing fresh feedback: motor " + std::to_string(i + 1));
      if (status[i] != required_status) throw std::runtime_error("Unexpected motor state/fault: motor " + std::to_string(i + 1) + " status=" + std::to_string(status[i]));
      if (!std::isfinite(position[i]) || std::abs(velocity[i]) > .15)
        throw std::runtime_error("Motor not stationary for hold startup: " + std::to_string(i + 1));
    }
  }
};
}  // namespace openarm_hardware
