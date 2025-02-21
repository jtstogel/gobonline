#ifndef SERIALPORT_H
#define SERIALPORT_H

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace gobonline {

class SerialPort {
 public:
  ~SerialPort();
  SerialPort(const SerialPort&) = delete;

  static absl::StatusOr<std::unique_ptr<SerialPort>> Open(std::string port_name);

  void Close();

  absl::Status Write(const std::string& data);

  absl::Status ReadLine(char dst[], size_t size);

  bool IsOpen() const;

 private:
  explicit SerialPort(std::string port_name);
  absl::Status Connect();

  int GetCharacter();

  std::string port_name_;
  int fd_;
  bool open_;

  static constexpr size_t kBufSize = BUFSIZ;
  char buf_[kBufSize];
  size_t buf_head_;
  size_t buf_tail_;
};

}  // namespace gobonline

#endif  // SERIALPORT_H
