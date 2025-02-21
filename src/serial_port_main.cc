#include <memory>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "src/serial_port.h"
#include "src/util/status.h"

ABSL_FLAG(std::string, port, "", "Serial port to open.");

using gobonline::SerialPort;

absl::Status Ping() {
  std::string port_name = absl::GetFlag(FLAGS_port);

  ASSIGN_OR_RETURN(std::unique_ptr<SerialPort> port,
                   SerialPort::Open(absl::GetFlag(FLAGS_port)));

  std::string cmd = "cmd:1234:ping\n";
  RETURN_IF_ERROR(port->Write(cmd.c_str()));
  std::cout << "send: " << cmd;

  char buf[256];
  RETURN_IF_ERROR(port->ReadLine(buf, sizeof(buf)));
  std::cout << "recieved: " << buf << std::endl;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  absl::Status s = Ping();
  if (!s.ok()) {
    std::cerr << s << std::endl;
    return 1;
  }

  return 0;
}
