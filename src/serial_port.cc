#include "serial_port.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace gobonline {

SerialPort::SerialPort(const std::string& port_name, int baud_rate)
    : port_name_(port_name),
      baud_rate_(baud_rate),
      serial_fd_(-1),
      open_(false) {}

SerialPort::~SerialPort() { Close(); }

absl::StatusOr<std::unique_ptr<SerialPort>> SerialPort::Open(
    const std::string& port_name, int baud_rate) {
  auto port = std::unique_ptr<SerialPort>(new SerialPort(port_name, baud_rate));
  absl::Status s = port->Connect();
  if (!s.ok()) {
    return s;
  }
  return std::move(port);
}

absl::Status SerialPort::Connect() {
  serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY);
  if (serial_fd_ == -1) {
    return absl::InternalError(
        absl::StrCat("failed to open serial port: ", port_name_));
  }

  struct termios tty;
  std::memset(&tty, 0, sizeof tty);
  if (tcgetattr(serial_fd_, &tty) != 0) {
    close(serial_fd_);
    return absl::InternalError(
        absl::StrCat("failed to get port attributes for port: ", port_name_));
  }

  cfsetospeed(&tty, B9600);
  cfsetispeed(&tty, B9600);
  tty.c_cflag = CS8 | CREAD | CLOCAL;
  tty.c_iflag = IGNPAR;
  tty.c_oflag = 0;
  tty.c_lflag = 0;

  tty.c_cc[VTIME] = 10;
  tty.c_cc[VMIN] = 0;

  tcflush(serial_fd_, TCIFLUSH);
  tcsetattr(serial_fd_, TCSANOW, &tty);

  open_ = true;
  return absl::OkStatus();
}

void SerialPort::Close() {
  if (open_) {
    close(serial_fd_);
    open_ = false;
  }
}

absl::Status SerialPort::Write(const std::string& data) {
  if (!open_) {
    return absl::InternalError("SerialPort::Write failed: port is closed");
  }
  size_t n = write(serial_fd_, data.c_str(), data.size());
  if (n < data.size()) {
    return absl::InternalError(
        "SerialPort::Write failed: failed to write all data");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> SerialPort::ReadUntil(char terminator) {
  if (!open_) {
    return absl::InternalError("SerialPort::Write failed: port is closed");
  }

  char buffer[256];
  std::memset(buffer, '\0', sizeof(buffer));
  int bytes_read = read(serial_fd_, buffer, sizeof(buffer) - 1);

  if (bytes_read > 0) {
    buffer[bytes_read] = '\0';
    return std::string(buffer);
  }

  return "";
}

bool SerialPort::IsOpen() const { return open_; }

}  // namespace gobonline
