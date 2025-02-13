#include "src/laser_calibration_solver.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "LBFGS.h"
#include "LBFGSpp/Param.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "autodiff/reverse/var.hpp"
#include "autodiff/reverse/var/eigen.hpp"
#include "src/util/status.h"

namespace gobonline {

namespace {

using autodiff::reverse::detail::cos;
using autodiff::reverse::detail::sin;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

// Maximum number of times to attempt optimizion
// the board's location before giving up.
constexpr double kMaxMinimizationAttempts = 1;

// The maximum allowable distance between the
// laser's projection and the desired spot on the board.
constexpr double kMaxLaserErrorMillimeters = 4.;

constexpr double kMillimetersPerFoot = 304.8;

};  // namespace

absl::Status CheckMirrorAngleBounds(double angle_radians) {
  if (angle_radians <= 0 || angle_radians >= std::numbers::pi / 2) {
    return absl::InvalidArgumentError(
        "mirror angle must be between 0 and pi/2");
  }
  return absl::OkStatus();
}

template <typename T, int Size>
Eigen::Vector<T, Size> CrossProduct(const Eigen::Vector<T, Size>& a,
                                    const Eigen::Vector<T, Size>& b) {
  return Eigen::Vector<T, Size>(a[1] * b[2] - a[2] * b[1],
                                a[2] * b[0] - a[0] * b[2],
                                a[0] * b[1] - a[1] * b[2]);
}

absl::StatusOr<LaserCalibrationSample> SimulateLaserPath(
    const MirrorAngles& angles, const BoardLocationAndOrientation& board) {
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.first_mirror_angle_radians));
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.second_mirror_angle_radians));

  double m1 = angles.first_mirror_angle_radians;
  double m2 = angles.second_mirror_angle_radians;

  Vector3d laser_dir = {
      std::cos(2 * m1),
      std::sin(2 * m1) * sin(2 * m2),
      -1 * std::sin(2 * m1) * std::cos(2 * m2),
  };
  laser_dir /= std::sqrt(laser_dir.dot(laser_dir));

  Vector3d laser_offset = {
      kMirrorDistanceMillimeters / std::tan(2 * m1),
      0,
      kMirrorDistanceMillimeters,
  };

  Vector3d board_normal = board.normal;
  double td = board_normal.dot(laser_dir);
  if (std::abs(td) < 0.001) {
    return absl::InvalidArgumentError(
        "failed to simulate laser path: board is coplaner with laser.");
  }
  double t = (board.center_offset - laser_offset).dot(board_normal) / td;

  Vector3d intersection = laser_offset + t * laser_dir;
  return LaserCalibrationSample{
      .pos = Vector2d((intersection - board.center_offset).dot(board.x_axis),
                      (intersection - board.center_offset)
                          .dot(CrossProduct(board.normal, board.x_axis))),
      .mirror_angles = angles,
  };
}

struct LaserSimulationErrorParams {
  // Offset of the board relative to the center of the first mirror.
  autodiff::Vector3var board_offset;

  // The board's orientation.
  //
  // board_x and board_normal are together the Gram-Schmidt
  // orthonormalization encoding of a rotation matrix.
  //
  // TODO(jtstogel): consider using the R^9 SVD representation,
  // as in "Learning with 3D rotations, a hitchhiker's guide to SO(3)"
  // https://arxiv.org/pdf/2404.11735v1
  autodiff::Vector3var board_normal;
  autodiff::Vector3var board_x;

  autodiff::Vector2var sample;

  autodiff::var mirror_1_radians;
  autodiff::var mirror_2_radians;
};

autodiff::var LaserSimulationError(const LaserSimulationErrorParams& p) {
  using autodiff::reverse::detail::cos;
  using autodiff::reverse::detail::sin;
  using autodiff::reverse::detail::tan;

  autodiff::Vector3var board_origin = p.board_offset;
  autodiff::Vector3var board_normal = p.board_normal.normalized();
  autodiff::Vector3var board_x =
      (p.board_x - p.board_x.dot(board_normal) * board_normal).normalized();
  autodiff::Vector3var board_y = CrossProduct(board_normal, board_x);

  autodiff::var m1 = p.mirror_1_radians;
  autodiff::var m2 = p.mirror_2_radians;
  autodiff::Vector3var laser_dir(3);
  laser_dir << cos(2 * m1), sin(2 * m1) * sin(2 * m2),
      -1 * sin(2 * m1) * cos(2 * m2);
  laser_dir = laser_dir.normalized();

  autodiff::Vector3var laser_offset(3);
  laser_offset << kMirrorDistanceMillimeters / tan(2 * m1), 0,
      kMirrorDistanceMillimeters;

  autodiff::var t = (board_origin - laser_offset).dot(board_normal) /
                    laser_dir.dot(board_normal);
  autodiff::Vector3var intersection = laser_offset + t * laser_dir;

  autodiff::Vector2var computed_location(
      (intersection - board_origin).dot(board_x),
      (intersection - board_origin).dot(board_y));

  // Penalize solutions that are too far away from the camera.
  // Any solution than 10ft away is bad.
  constexpr double kMaxUnpenalizedDistance = 10 * kMillimetersPerFoot;
  autodiff::var board_distance = board_origin.norm();
  autodiff::var board_distance_penalty = autodiff::reverse::detail::condition(
      board_distance > kMaxUnpenalizedDistance,
      autodiff::reverse::detail::pow(board_distance - kMaxUnpenalizedDistance,
                                     2),
      0);

  return (computed_location - p.sample).squaredNorm() + board_distance_penalty;
}

/**
 * Finds the spacing of a 1D grid such that the sum of squared distances from
 * the points to the grid lines is minimized.
 */
class BoardLocationProblem {
 public:
  explicit BoardLocationProblem(
      absl::Span<const LaserCalibrationSample> samples)
      : samples_(samples) {}

  double operator()(const VectorXd& x, VectorXd& grad) const {
    autodiff::VectorXvar v = x;
    autodiff::var error = 0;
    for (const LaserCalibrationSample& sample : samples_) {
      error += LaserSimulationError(LaserSimulationErrorParams{
          .board_offset = autodiff::Vector3var(v[0], v[1], v[2]),
          .board_normal = autodiff::Vector3var(v[3], v[4], v[5]),
          .board_x = autodiff::Vector3var(v[6], v[7], v[8]),
          .sample = sample.pos,
          .mirror_1_radians = sample.mirror_angles.first_mirror_angle_radians,
          .mirror_2_radians = sample.mirror_angles.second_mirror_angle_radians,
      });
    }
    auto derivatives = autodiff::derivatives(
        error, autodiff::reverse::detail::wrt(v[0], v[1], v[2], v[3], v[4],
                                              v[5], v[6], v[7], v[8]));
    for (int i = 0; i < x.size(); i++) {
      grad[i] = derivatives[i];
    }
    return autodiff::val(error);
  }

  static BoardLocationAndOrientation Unpack(const VectorXd& x) {
    Vector3d board_normal(x[3], x[4], x[5]);
    board_normal = board_normal.normalized();
    Vector3d board_x_axis(x[6], x[7], x[8]);
    board_x_axis -= board_x_axis.dot(board_normal) * board_normal;
    board_x_axis = board_x_axis.normalized();

    return BoardLocationAndOrientation{
        .center_offset = Vector3d(x[0], x[1], x[2]),
        .normal = board_normal,
        .x_axis = board_x_axis,
    };
  }

 private:
  absl::Span<const LaserCalibrationSample> samples_;
};

std::ostream& operator<<(std::ostream& os,
                         const BoardLocationAndOrientation& loc) {
  os << "BoardLocationAndOrientation{"
     << ".center_offset=" << loc.center_offset.transpose()
     << ", .normal=" << loc.normal.transpose()
     << ", .x_axis=" << loc.x_axis.transpose() << "}";
  return os;
}

Eigen::VectorXd RandomVector(const Eigen::VectorXd& lower_bound,
                             const Eigen::VectorXd& upper_bound,
                             absl::BitGen& gen) {
  assert(lower_bound.size() == upper_bound.size());

  VectorXd x = VectorXd::Zero(lower_bound.size());
  for (int i = 0; i < x.size(); i++) {
    x[i] = absl::Uniform(absl::IntervalClosed, gen, lower_bound[i],
                         upper_bound[i]);
  }

  return x;
}

absl::StatusOr<std::pair<BoardLocationAndOrientation, double>>
ComputeBoardLocationWithError(
    absl::Span<const LaserCalibrationSample> samples) {
  LBFGSpp::LBFGSParam<double> error_param;
  error_param.max_iterations = 1000;

  LBFGSpp::LBFGSSolver<double> solver(error_param);
  BoardLocationProblem f(samples);

  // This is in mm, so allots for the Goban being 1ft-10ft away.
  VectorXd lb = VectorXd::Constant(9, -5 * kMillimetersPerFoot);
  VectorXd ub = VectorXd::Constant(9, 5 * kMillimetersPerFoot);

  lb[1] = 1. * kMillimetersPerFoot;
  ub[1] = 10. * kMillimetersPerFoot;

  absl::BitGen gen;
  Eigen::VectorXd x = RandomVector(lb, ub, gen);
  Eigen::VectorXd initial_x(x);
  BoardLocationAndOrientation initial = BoardLocationProblem::Unpack(x);

  double error = std::numeric_limits<double>::max();
  try {
    int iters = solver.minimize(f, x, error);

    // The Goban will always be set in the positive y-direction,
    // since the board must be in front of the camera.
    // If the solver found a solution such that the board intersects
    // with the laser in the direction opposite to the camera,
    // then we just flip the solution.
    if (x[1] < 0) {
      x *= -1;
    }

    std::cerr << (error < 1 ? "good:" : "bad:")                          //
              << "\n  Error: " << error                                  //
              << "\n  Iterations: " << iters                             //
              << "\n  Initial X: " << initial_x.transpose()              //
              << "\n  Initial board: " << initial                        //
              << "\n  Final board: " << BoardLocationProblem::Unpack(x)  //
              << "\n  FinalGrad: " << solver.final_grad().transpose()    //
              << "\n  FinalGradNorm: " << solver.final_grad_norm()       //
              << std::endl;

  } catch (std::runtime_error& e) {
    return absl::InternalError(absl::StrCat("failed to minimize: ", e.what()));
  }
  return std::make_pair(BoardLocationProblem::Unpack(x), error);
}

absl::StatusOr<BoardLocationAndOrientation> ComputeBoardLocation(
    absl::Span<const LaserCalibrationSample> samples) {
  double error = std::numeric_limits<double>::max();
  absl::StatusOr<BoardLocationAndOrientation> loc =
      absl::InternalError("failed to find any board locations");

  for (int attempts = 0; attempts < kMaxMinimizationAttempts; attempts++) {
    auto res = ComputeBoardLocationWithError(samples);
    if (!res.ok()) {
      if (!loc.ok()) {
        loc.AssignStatus(res.status());  // Propagate the last error.
      }
      continue;
    }

    if (res->second < error) {
      error = res->second;
      loc.emplace(std::move(res->first));
    }

    if (error < kMaxLaserErrorMillimeters) {
      return *loc;
    }
  }

  if (error >= kMaxLaserErrorMillimeters) {
    return absl::InternalError(absl::StrCat(
        "failed to find a good board location, best error=", error));
  }

  return loc;
}

}  // namespace gobonline
