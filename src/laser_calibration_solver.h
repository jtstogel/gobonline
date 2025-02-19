#ifndef SRC_LASER_CALIBRATION_SOLVER_H
#define SRC_LASER_CALIBRATION_SOLVER_H

#include <Eigen/Core>

#include "absl/random/bit_gen_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace gobonline {

/**
 * Distance between the center of the laser's mirrors.
 */
inline constexpr double kMirrorDistanceMillimeters = 25.;

/** Millimeters in a foot. */
constexpr double kMmPerFoot = 304.8;

struct MirrorAngles {
  // The angle the first mirror took, relative to the x-y plane.
  double first_mirror_angle_radians;

  // The angle the second mirror took, relative to the x-y plane.
  double second_mirror_angle_radians;
};

struct LaserPositionOnBoard {
  // A point on the board, relative to its center.
  // Units are in millimeters.
  Eigen::Vector2d position;
};

struct BoardLocationAndOrientation {
  // Position of the board's center,
  // relative to the center of the first mirror.
  Eigen::Vector3d origin_offset;

  // Direction vector for the x-axis of the board.
  // Defines the board's plane when taken with y_axis.
  Eigen::Vector3d x_axis;

  // Direction vector for the y-axis of the board.
  // Defines the board's plane when taken with x_axis.
  Eigen::Vector3d y_axis;
};

/**
 * Computes the position of the laser on the board given mirror angles.
 */
absl::StatusOr<LaserPositionOnBoard> ComputeLaserPositionOnBoard(
    const MirrorAngles& angles, const BoardLocationAndOrientation& board);

/**
 * Computes the mirror angles required to point the laser at a board position.
 */
absl::StatusOr<MirrorAngles> ComputeLaserMirrorAngles(
    absl::BitGenRef gen, const BoardLocationAndOrientation& board,
    const LaserPositionOnBoard& position);

struct LaserCalibrationSample {
  LaserPositionOnBoard position;
  MirrorAngles mirror_angles;
};

/**
 * Computes the board's location given a list of measured samples.
 */
absl::StatusOr<BoardLocationAndOrientation> ComputeBoardLocation(
    absl::BitGenRef gen, absl::Span<const LaserCalibrationSample> samples,
    int attempts = 3);

}  // namespace gobonline

#endif
