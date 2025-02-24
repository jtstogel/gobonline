#pragma once

#include <cstdio>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/serial_port.h"

namespace gobonline {

class ArduinoController {
 public:
  explicit ArduinoController(std::unique_ptr<SerialPort> port)
      : port_(std::move(port)) {}

  absl::Status Ping();

  absl::Status SetMirrorPositions(const std::array<int32_t, 2>& positions);

  absl::StatusOr<std::array<int32_t, 2>> GetMirrorPositions();

 private:
  int SequenceNo();

  int seq_ = 0;
  std::unique_ptr<SerialPort> port_;
};

}  // namespace gobonline
