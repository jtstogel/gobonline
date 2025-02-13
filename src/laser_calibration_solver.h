#ifndef SRC_LASER_CALIBRATION_SOLVER_H
#define SRC_LASER_CALIBRATION_SOLVER_H

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include <Eigen/Core>

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
  Eigen::Vector2d pos;

  // The angles that the mirror
  MirrorAngles mirror_angles;
};

struct BoardLocationAndOrientation {
  // Position of the board's center,
  // relative to the center of the first mirror.
  Eigen::Vector3d center_offset;

  // Direction vector for the z-axis of the board.
  // Defines the board's plane when taken with board_x_axis.
  Eigen::Vector3d normal;

  // Direction vector for the y-axis of the board.
  // Defines the board's plane when taken with board_x_axis.
  Eigen::Vector3d x_axis;
};

absl::StatusOr<LaserCalibrationSample> SimulateLaserPath(
    const MirrorAngles& angles,
    const BoardLocationAndOrientation& board);

absl::StatusOr<BoardLocationAndOrientation> ComputeBoardLocation(
    absl::Span<const LaserCalibrationSample> samples);

}  // namespace gobonline

#endif
