#include "src/arduino_controller.h"

#include <cstdio>
#include <memory>
#include <regex>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "re2/re2.h"
#include "src/serial_port.h"
#include "src/util/status.h"

namespace gobonline {
namespace {

const LazyRE2 kPongRe = {R"(res:([0-9]{4}):pong)"};

const LazyRE2 kGetPositionsRe = {
    R"(res:([0-9]{4}):get_positions:([0-9]+),([0-9]+))"};

const LazyRE2 kSetPositionsRe = {R"(res:([0-9]{4}):set_positions)"};

}  // namespace

absl::StatusOr<std::array<int32_t, 2>> ArduinoController::GetMirrorPositions() {
  int seq = SequenceNo();

  char cmd[128];
  snprintf(cmd, sizeof(cmd), "cmd:%04d:get_positions\n", seq);
  RETURN_IF_ERROR(port_->Write(cmd));

  char response[128];
  RETURN_IF_ERROR(port_->ReadLine(response, sizeof(response)));
  absl::string_view s(response);

  int response_seq;
  std::array<int32_t, 2> positions;
  if (!RE2::FullMatch(response, *kGetPositionsRe, &response_seq, &positions[0],
                      &positions[1])) {
    return absl::InternalError(
        absl::StrCat("Unexpected response from Arduino: ", response));
  }

  if (response_seq != seq) {
    return absl::InternalError(
        absl::StrCat("Unexpected sequence number from Arduino: ", response));
  }

  return positions;
}

absl::Status ArduinoController::SetMirrorPositions(
    const std::array<int32_t, 2>& positions) {
  int seq = SequenceNo();

  char cmd[128];
  snprintf(cmd, sizeof(cmd), "cmd:%04d:set_positions:%d,%d\n", seq,
           positions[0], positions[1]);
  RETURN_IF_ERROR(port_->Write(cmd));

  char response[128];
  RETURN_IF_ERROR(port_->ReadLine(response, sizeof(response)));
  absl::string_view s(response);

  int response_seq;
  if (!RE2::FullMatch(response, *kSetPositionsRe, &response_seq)) {
    return absl::InternalError(
        absl::StrCat("Unexpected response from Arduino: ", response));
  }

  if (response_seq != seq) {
    return absl::InternalError(
        absl::StrCat("Unexpected sequence number from Arduino: ", response));
  }

  return absl::OkStatus();
}

absl::Status ArduinoController::Ping() {
  int seq = SequenceNo();

  char cmd[128];
  snprintf(cmd, sizeof(cmd), "cmd:%04d:ping\n", seq);
  RETURN_IF_ERROR(port_->Write(cmd));

  char response[128];
  RETURN_IF_ERROR(port_->ReadLine(response, sizeof(response)));
  absl::string_view s(response);

  int response_seq;
  if (!RE2::FullMatch(response, *kPongRe, &response_seq)) {
    return absl::InternalError(
        absl::StrCat("Unexpected response from Arduino: ", response));
  }

  if (response_seq != seq) {
    return absl::InternalError(
        absl::StrCat("Unexpected sequence number from Arduino: ", response));
  }

  return absl::OkStatus();
}

int ArduinoController::SequenceNo() {
  seq_ = (seq_ + 1) % 10000;
  return seq_;
}

}  // namespace gobonline
