#ifndef SRC_LASER_CALIBRATION_SOLVER_H
#define SRC_LASER_CALIBRATION_SOLVER_H

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "opencv2/core/types.hpp"

namespace gobonline {

/**
 * Distance between the center of the laser's mirrors.
 */
inline constexpr double kMirrorDistanceMillimeters = 25.;

struct MirrorAngles {
  // The angle the first mirror took, relative to the x-y plane.
  double first_mirror_angle_radians;

  // The angle the second mirror took, relative to the x-y plane.
  double second_mirror_angle_radians;
};

struct LaserCalibrationSample {
  // A point measured on the board, relative to its center.
  cv::Point2d pos;

  // The angles that the mirror
  MirrorAngles mirror_angles;
};

struct BoardLocationAndOrientation {
  // Position of the board's center,
  // relative to the center of the first mirror.
  cv::Vec3d center_offset;

  // Direction vector for the x-axis of the board.
  // Defines the board's plane when taken with board_y_axis.
  cv::Vec3d x_axis_dir;

  // Direction vector for the y-axis of the board.
  // Defines the board's plane when taken with board_x_axis.
  cv::Vec3d y_axis_dir;
};

absl::StatusOr<LaserCalibrationSample> SimulateLaserPath(
    const MirrorAngles& angles,
    const BoardLocationAndOrientation& board);

absl::StatusOr<BoardLocationAndOrientation> ComputeBoardLocation(
    absl::Span<LaserCalibrationSample> samples);

}  // namespace gobonline

#endif
