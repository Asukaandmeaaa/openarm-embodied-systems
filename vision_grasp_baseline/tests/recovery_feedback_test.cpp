#include "openarm_hardware/recovery_feedback.hpp"
#include <cassert>
#include <iostream>
using openarm_hardware::RecoveryFeedback;

bool rejects(const RecoveryFeedback& f, unsigned status) {
  try { f.validate(status); return false; } catch (const std::runtime_error&) { return true; }
}

int main() {
  RecoveryFeedback f;
  assert(rejects(f, 0)); // never substitute default zeros for missing motors
  for (unsigned i = 0; i < 8; ++i) {
    canfd_frame raw{}; raw.can_id = 0x11 + i; raw.len = 8;
    raw.data[0] = i+1; raw.data[1] = 0x83; raw.data[2] = 0x97;
    raw.data[3] = 0x7f; raw.data[4] = 0xf8;
    f.accept(raw);
  }
  f.validate(0);
  assert(std::abs(f.position[3] - .3507667658503095) < 1e-10);
  assert(rejects(f, 1)); // disabled is not enabled
  f.status[3] = 12; assert(rejects(f, 0)); // reported 0xC blocks activation
  f.status.fill(1); f.validate(1);
  f.velocity[2] = .5; assert(rejects(f, 1));
  f.velocity[2] = 0; f.seen[6] = false; assert(rejects(f, 1));
  canfd_frame bad{}; bad.can_id = 0x17; bad.len = 8; bad.data[0] = 8;
  f.accept(bad); assert(!f.seen[6]); // motor ID must match arbitration ID
  std::cout << "Recovery feedback regression passed; no CAN devices opened.\n";
}
