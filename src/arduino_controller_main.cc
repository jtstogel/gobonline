#include <memory>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/arduino_controller.h"
#include "src/serial_port.h"
#include "src/util/status.h"

ABSL_FLAG(std::string, port, "", "Serial port to open.");

namespace {

using gobonline::ArduinoController;
using gobonline::SerialPort;

absl::Status Ping() {
  std::string port_name = absl::GetFlag(FLAGS_port);

  ASSIGN_OR_RETURN(std::unique_ptr<SerialPort> port,
                   SerialPort::Open(absl::GetFlag(FLAGS_port)));

  ArduinoController controller(std::move(port));
  RETURN_IF_ERROR(controller.Ping());
  using Positions = std::array<int32_t, 2>;
  ASSIGN_OR_RETURN(Positions positions, controller.GetMirrorPositions());

  std::cout << "GetMirrorPositions: " << positions[0] << ", " << positions[1]
            << std::endl;

  int step = 2000;
  positions[0] += step;
  positions[1] += step;
  RETURN_IF_ERROR(controller.SetMirrorPositions(positions));

  ASSIGN_OR_RETURN(positions, controller.GetMirrorPositions());
  std::cout << "GetMirrorPositions: " << positions[0] << ", " << positions[1]
            << std::endl;

  positions[0] -= step;
  positions[1] -= step;
  RETURN_IF_ERROR(controller.SetMirrorPositions(positions));

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  absl::Status s = Ping();
  if (!s.ok()) {
    std::cout << s << std::endl;
    return 1;
  }

  return 0;
}
