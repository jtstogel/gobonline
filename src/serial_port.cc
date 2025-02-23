#include "serial_port.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/util/status.h"

namespace gobonline {

SerialPort::SerialPort(std::string port_name)
    : port_name_(std::move(port_name)), fd_(-1), open_(false) {}

SerialPort::~SerialPort() { Close(); }

absl::StatusOr<std::unique_ptr<SerialPort>> SerialPort::Open(
    std::string port_name) {
  auto port = std::unique_ptr<SerialPort>(new SerialPort(std::move(port_name)));
  RETURN_IF_ERROR(port->Connect());
  return std::move(port);
}

absl::Status SerialPort::Connect() {
  fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY);
  if (fd_ == -1) {
    return absl::InternalError(
        absl::StrCat("failed to open serial port: ", port_name_));
  }

  struct termios tty;
  std::memset(&tty, 0, sizeof tty);
  if (tcgetattr(fd_, &tty) != 0) {
    close(fd_);
    return absl::InternalError(
        absl::StrCat("failed to get port attributes for port: ", port_name_));
  }

  cfsetospeed(&tty, B9600);
  cfsetispeed(&tty, B9600);
  tty.c_cflag = CS8 | CREAD | CLOCAL;
  tty.c_iflag = IGNPAR;
  tty.c_oflag = 0;
  tty.c_lflag = 0;

  tty.c_cc[VTIME] = 100;
  tty.c_cc[VMIN] = 0;

  tcflush(fd_, TCIFLUSH);
  tcsetattr(fd_, TCSANOW, &tty);

  open_ = true;
  return absl::OkStatus();
}

void SerialPort::Close() {
  if (open_) {
    close(fd_);
    open_ = false;
  }
}

absl::Status SerialPort::Write(std::string_view data) const {
  if (!open_) {
    return absl::InternalError("SerialPort::Write failed: port is closed");
  }

  size_t n = write(fd_, data.data(), data.size());
  if (n < data.size()) {
    return absl::InternalError(
        "SerialPort::Write failed: failed to write data");
  }

  return absl::OkStatus();
}

int SerialPort::GetCharacter() {
  if (buf_head_ == buf_tail_) {
    int n = read(fd_, buf_, kBufSize);
    if (n <= 0) {
      return n;
    }

    buf_head_ = 0;
    buf_tail_ = n;
  }

  return buf_[buf_head_++];
}

absl::Status SerialPort::ReadLine(char dst[], size_t size) {
  if (!open_) {
    return absl::InternalError("ReadUntil failed: port is closed");
  }

  size_t n = 0;
  while (n < size - 1) {
    int ch = GetCharacter();
    if (ch == -1) {
      return absl::InternalError(
          absl::StrCat("ReadUntil failed: read: ", strerror(errno)));
    }

    if (ch == '\n') {
      dst[n] = '\0';
      return absl::OkStatus();
    }

    if (ch == 0) {
      return absl::InternalError("ReadUntil failed: timeout");
    }

    dst[n++] = ch;
  }

  return absl::InternalError(
      "ReadUntil failed: not enough space in destination buffer");
}

bool SerialPort::IsOpen() const { return open_; }

}  // namespace gobonline
