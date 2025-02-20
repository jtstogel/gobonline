#ifndef SERIALPORT_H
#define SERIALPORT_H

#include <string>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace gobonline {

class SerialPort {
 public:
  ~SerialPort();
  SerialPort(const SerialPort&) = delete;

  absl::StatusOr<std::unique_ptr<SerialPort>> Open(const std::string& port_name,
                                                   int baud_rate = 9600);
  void Close();

  absl::Status Write(const std::string& data);

  std::string ReadUntil(char terminator = '\n');

  bool IsOpen() const;

 private:
  explicit SerialPort(const std::string& port_name, int baud_rate = 9600);
  absl::Status Connect();

  std::string port_name_;
  int baud_rate_;
  int serial_fd_;
  bool open_;
};

}  // namespace gobonline

#endif  // SERIALPORT_H
