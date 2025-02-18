#include "src/laser_calibration_solver.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <cmath>
#include <numbers>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/util/simanneal/simanneal.h"
#include "src/util/status.h"

namespace gobonline {

namespace {

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

// The maximum allowable distance between the
// laser's projection and the desired spot on the board.
constexpr double kMaxLaserErrorMm = 4.;

template <size_t N>
VectorXd VecFromArray(const std::array<double, N>& array) {
  VectorXd v(array.size());
  for (size_t i = 0; i < array.size(); i++) {
    v[i] = array[i];
  }
  return v;
}

struct VectorLine {
  Vector3d direction;
  Vector3d point_on_line;
};

struct VectorPlane {
  Vector3d normal;
  Vector3d point_on_plane;
};

VectorLine LaserLine(const MirrorAngles& angles) {
  double m1 = angles.first_mirror_angle_radians;
  double m2 = angles.second_mirror_angle_radians;
  Eigen::Vector3d laser_dir{
      std::cos(2 * m1),
      std::sin(2 * m1) * std::sin(2 * m2),
      -1 * std::sin(2 * m1) * std::cos(2 * m2),
  };
  laser_dir.normalize();

  Eigen::Vector3d laser_offset{
      kMirrorDistanceMillimeters / std::tan(2 * m1),
      0,
      kMirrorDistanceMillimeters,
  };
  return {
      .direction = laser_dir,
      .point_on_line = laser_offset,
  };
}

Vector3d LinePlaneIntesection(const VectorLine& line,
                              const VectorPlane& plane) {
  double td = line.direction.dot(plane.normal);

  static constexpr double kMinCorrelation = 1e-8;
  if (std::fabs(td) < kMinCorrelation) {
    td = (std::signbit(td) ? -1. : 1.) * kMinCorrelation;
  }

  double t = (plane.point_on_plane - line.point_on_line).dot(plane.normal) / td;
  return line.point_on_line + t * line.direction;
}

/**
 * An overly specified board location and orientation.
 * Used to avoid recomputing values twice.
 */
struct BoardSpecification {
  Vector3d origin;
  Vector3d x_axis;
  Vector3d y_axis;
  Vector3d z_axis;
};

/**
 * Simulates the projection of a laser onto the plane of the board.
 * Returns the laser's 2D coordinates in terms of the board's origin.
 */
Vector2d SimulateLaserLocationOnBoard(const BoardSpecification& board,
                                      const VectorLine& laser_line) {
  const VectorPlane board_plane = {
      .normal = board.z_axis,
      .point_on_plane = board.origin,
  };
  Vector3d intersection = LinePlaneIntesection(laser_line, board_plane);

  Vector3d location_relative_to_board_origin = intersection - board.origin;
  return {
      location_relative_to_board_origin.dot(board.x_axis),
      location_relative_to_board_origin.dot(board.y_axis),
  };
}

class BoardLocationSolver {
 private:
  // A version of LaserCalibrationSample with some precomputation done.
  struct Sample {
    LaserPositionOnBoard position_on_board;
    VectorLine laser_line;
  };

  static constexpr size_t kDims = 9;

 public:
  static absl::StatusOr<BoardLocationAndOrientation> Solve(
      absl::BitGenRef gen, absl::Span<const LaserCalibrationSample> samples) {
    BoardLocationSolver problem(samples);

    using Optimizer = simanneal::SimulatedAnnealingOptimizer<kDims>;
    Optimizer optimizer(Optimizer::Config{
        .bounds = kBounds,
        .initial_temperature = 500000,
        .max_iterations = 100000,
    });

    ASSIGN_OR_RETURN(
        Optimizer::Result result,
        optimizer.Minimize(gen, [&problem](const std::array<double, kDims>& x) {
          return problem.Error(x);
        }));

    if (result.error > std::pow(kMaxLaserErrorMm, 2)) {
      return absl::InternalError(absl::StrCat(
          "Failed to find a good board location, error=", result.error,
          " is greater than limit=", std::pow(kMaxLaserErrorMm, 2)));
    }

    BoardSpecification board = UnpackBoardParameters(VecFromArray(result.x));
    return BoardLocationAndOrientation{
        .origin_offset = board.origin,
        .x_axis = board.x_axis,
        .y_axis = board.y_axis,
    };
  }

 private:
  explicit BoardLocationSolver(
      absl::Span<const LaserCalibrationSample> samples) {
    for (const LaserCalibrationSample& s : samples) {
      samples_.push_back(Sample{
          .position_on_board = s.position,
          .laser_line = LaserLine(s.mirror_angles),
      });
    }
  }

  double Error(const std::array<double, kDims>& x_array) {
    VectorXd v = VecFromArray(x_array);

    BoardSpecification board = UnpackBoardParameters(v);

    double error = 0.;
    for (const Sample& s : samples_) {
      Vector2d computed_location =
          SimulateLaserLocationOnBoard(board, s.laser_line);
      error += (computed_location - s.position_on_board.position).squaredNorm();
    }

    return error / samples_.size();
  }

  static BoardSpecification UnpackBoardParameters(const VectorXd& x) {
    BoardSpecification board;

    board.origin = x.segment(0, 3);

    // z_axis and x_axis together represent the orientation of the board.
    // Gram schmidt orthonormalization is performed below to get properly
    // aligned axes from the unconstrained variables in `x`.
    //
    // Consider using a more friendly representation:
    // https://arxiv.org/pdf/2404.11735v1
    board.z_axis = x.segment(3, 6);
    board.z_axis.normalize();

    board.x_axis = x.segment(6, 9);
    board.x_axis -= board.x_axis.dot(board.z_axis) * board.z_axis;
    board.x_axis.normalize();

    board.y_axis = board.z_axis.cross(board.x_axis);

    return board;
  }

  static constexpr std::array<std::array<double, 2>, kDims> kBounds = {{
      // Bounds for the position of the board.
      // The y-position is strictly positive,
      // since the board must be in front of the camera.
      {-5 * kMmPerFoot, 5 * kMmPerFoot},
      {1. * kMmPerFoot, 10 * kMmPerFoot},
      {-5 * kMmPerFoot, 5 * kMmPerFoot},

      // Bounds for the board's z-axis (the axis normal to its plane).
      // The y-coordinate is strictly negative,
      // since the board must be facing the camera.
      {-1., 1.},
      {-1., -1e-4},
      {-1., 1.},

      // Bounds for the board's x-axis.
      {-1., 1.},
      {-1., 1.},
      {-1., 1.},
  }};

  std::vector<Sample> samples_;
};

absl::Status CheckMirrorAngleBounds(double angle_radians) {
  if (angle_radians <= 0 || angle_radians >= std::numbers::pi / 2) {
    return absl::InvalidArgumentError(
        "mirror angle must be between 0 and pi/2");
  }
  return absl::OkStatus();
}

class MirrorAnglesSolver {
 private:
  static constexpr size_t kDims = 2;

 public:
  static absl::StatusOr<MirrorAngles> Solve(
      absl::BitGenRef gen, const BoardLocationAndOrientation& board,
      const LaserPositionOnBoard& position) {
    MirrorAnglesSolver problem(board, position);

    using Optimizer = simanneal::SimulatedAnnealingOptimizer<kDims>;
    Optimizer optimizer(Optimizer::Config{
        .bounds = kBounds,
        .initial_temperature = 50000,
        .max_iterations = 20000,
    });

    ASSIGN_OR_RETURN(
        Optimizer::Result result,
        optimizer.Minimize(gen, [&problem](const std::array<double, kDims>& x) {
          return problem.Error(x);
        }));

    if (result.error > 1e-2) {
      return absl::InternalError(absl::StrCat(
          "Failed to find good mirror angles, error=", result.error,
          " is greater than limit=1e-2"));
    }

    return UnpackMirrorAngles(result.x);
  }

 private:
  explicit MirrorAnglesSolver(const BoardLocationAndOrientation& board,
                              const LaserPositionOnBoard& position)
      : position_(position) {
    board_ = {
        .origin = board.origin_offset,
        .x_axis = board.x_axis,
        .y_axis = board.y_axis,
        .z_axis = board.x_axis.cross(board.y_axis),
    };
  }

  double Error(const std::array<double, kDims>& x) {
    MirrorAngles angles = UnpackMirrorAngles(x);

    Vector2d computed_location =
        SimulateLaserLocationOnBoard(board_, LaserLine(angles));
    return (computed_location - position_.position).squaredNorm();
  }

  static MirrorAngles UnpackMirrorAngles(const std::array<double, kDims>& x) {
    return {
        .first_mirror_angle_radians = x[0],
        .second_mirror_angle_radians = x[1],
    };
  }

  static constexpr std::array<std::array<double, 2>, kDims> kBounds = {{
      {1e-2, std::numbers::pi / 2 - 1e-2},
      {1e-2, std::numbers::pi / 2 - 1e-2},
  }};

  BoardSpecification board_;
  const LaserPositionOnBoard& position_;
};

};  // namespace

absl::StatusOr<LaserPositionOnBoard> ComputeLaserPositionOnBoard(
    const MirrorAngles& angles, const BoardLocationAndOrientation& board) {
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.first_mirror_angle_radians));
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.second_mirror_angle_radians));

  BoardSpecification board_spec = {
      .origin = board.origin_offset,
      .x_axis = board.x_axis,
      .y_axis = board.y_axis,
      .z_axis = board.x_axis.cross(board.y_axis),
  };
  VectorLine laser_line = LaserLine(angles);
  return LaserPositionOnBoard{
      .position = SimulateLaserLocationOnBoard(board_spec, laser_line),
  };
}

absl::StatusOr<MirrorAngles> ComputeLaserMirrorAngles(
    absl::BitGenRef gen, const BoardLocationAndOrientation& board,
    const LaserPositionOnBoard& position) {
  return MirrorAnglesSolver::Solve(gen, board, position);
}

absl::StatusOr<BoardLocationAndOrientation> ComputeBoardLocation(
    absl::BitGenRef gen, absl::Span<const LaserCalibrationSample> samples,
    int attempts) {
  while (true) {
    absl::StatusOr<BoardLocationAndOrientation> result =
        BoardLocationSolver::Solve(gen, samples);
    attempts--;
    if (!result.ok() && attempts > 0) {
      continue;
    }
    return result;
  }
}

}  // namespace gobonline
