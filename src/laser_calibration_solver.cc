#include "src/laser_calibration_solver.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "src/util/status.h"

namespace gobonline {

absl::Status CheckMirrorAngleBounds(double angle_radians) {
  if (angle_radians <= 0 || angle_radians >= std::numbers::pi / 2) {
    return absl::InvalidArgumentError(
        "mirror angle must be between 0 and pi/2");
  }
  return absl::OkStatus();
}

absl::StatusOr<LaserCalibrationSample> SimulateLaserPath(
    const MirrorAngles& angles, const BoardLocationAndOrientation& board) {
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.first_mirror_angle_radians));
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.second_mirror_angle_radians));

  double m1 = angles.first_mirror_angle_radians;
  double m2 = angles.second_mirror_angle_radians;

  cv::Vec3d laser_dir = {
      std::cos(2 * m1),
      std::sin(2 * m1) * sin(2 * m2),
      -1 * std::sin(2 * m1) * std::cos(2 * m2),
  };
  laser_dir /= std::sqrt(laser_dir.ddot(laser_dir));

  cv::Vec3d laser_offset = {
      kMirrorDistanceMillimeters / std::tan(2 * m1),
      0,
      kMirrorDistanceMillimeters,
  };

  cv::Vec3d board_normal = board.x_axis_dir.cross(board.y_axis_dir);
  std::cerr << "board_normal: " << board_normal[0] << "," << board_normal[1]
            << "," << board_normal[2] << std::endl;
  std::cerr << "laser: " << laser_dir[0] << "," << laser_dir[1] << ","
            << laser_dir[2] << std::endl;
  std::cerr << "laser.ddot(board_normal) = " << board_normal.ddot(laser_dir)
            << std::endl;

  double td = board_normal.ddot(laser_dir);
  if (std::abs(td) < 0.001) {
    return absl::InvalidArgumentError(
        "failed to simulate laser path: board is coplaner with laser.");
  }
  double t = (board.center_offset - laser_offset).ddot(board_normal) / td;

  cv::Vec3d intersection = laser_offset + t * laser_dir;
  std::cerr << "intersection: " << intersection[0] << "," << intersection[1]
            << "," << intersection[2] << std::endl;

  return LaserCalibrationSample{
      .pos = cv::Point2d(
          (intersection - board.center_offset).ddot(board.x_axis_dir),
          (intersection - board.center_offset).ddot(board.y_axis_dir)),
      .mirror_angles = angles,
  };
}

absl::StatusOr<BoardLocationAndOrientation> ComputeBoardLocation(
    absl::Span<LaserCalibrationSample> samples) {
  return absl::InvalidArgumentError("");
}

}  // namespace gobonline
