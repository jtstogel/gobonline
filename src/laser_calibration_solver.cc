#include "src/laser_calibration_solver.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <utility>

#include "absl/status/statusor.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/optim.hpp"
#include "opencv2/core/types.hpp"
#include "src/util/status.h"

namespace gobonline {

namespace {

constexpr double kLargeError = 1000000000;
constexpr double kMaxMinimizationAttempts = 100;

};  // namespace

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
  double td = board_normal.ddot(laser_dir);
  if (std::abs(td) < 0.001) {
    return absl::InvalidArgumentError(
        "failed to simulate laser path: board is coplaner with laser.");
  }
  double t = (board.center_offset - laser_offset).ddot(board_normal) / td;

  cv::Vec3d intersection = laser_offset + t * laser_dir;
  return LaserCalibrationSample{
      .pos = cv::Point2d(
          (intersection - board.center_offset).ddot(board.x_axis_dir),
          (intersection - board.center_offset).ddot(board.y_axis_dir)),
      .mirror_angles = angles,
  };
}

/**
 * Finds the spacing of a 1D grid such that the sum of squared distances from
 * the points to the grid lines is minimized.
 */
class ComputeBoardLocationProblem : public cv::MinProblemSolver::Function {
 public:
  explicit ComputeBoardLocationProblem(
      absl::Span<const LaserCalibrationSample> samples)
      : samples_(samples) {}

  static absl::StatusOr<BoardLocationAndOrientation> Unpack(const double* x) {
    BoardLocationAndOrientation loc;
    loc.center_offset[0] = x[0];
    loc.center_offset[1] = x[1];
    loc.center_offset[2] = x[2];

    loc.x_axis_dir[0] = x[3];
    loc.x_axis_dir[1] = x[4];
    loc.x_axis_dir[2] = x[5];
    if (cv::norm(loc.x_axis_dir) == 0) {
      return absl::InvalidArgumentError(
          "x-axis norm is zero during minimization");
    }
    loc.x_axis_dir /= cv::norm(loc.x_axis_dir);

    loc.y_axis_dir[0] = x[6];
    loc.y_axis_dir[1] = x[7];
    loc.y_axis_dir[2] = x[8];
    loc.y_axis_dir -= loc.y_axis_dir.ddot(loc.x_axis_dir) * loc.x_axis_dir;
    if (cv::norm(loc.y_axis_dir) == 0) {
      return absl::InvalidArgumentError(
          "y-axis norm is zero during minimization");
    }
    loc.y_axis_dir /= cv::norm(loc.y_axis_dir);

    return loc;
  }

  double calc(const double* x) const override {
    absl::StatusOr<BoardLocationAndOrientation> loc = Unpack(x);
    if (!loc.ok()) {
      return kLargeError;
    }
    double error = 0;
    for (const auto& sample : samples_) {
      absl::StatusOr<LaserCalibrationSample> computed =
          SimulateLaserPath(sample.mirror_angles, *loc);
      if (!computed.ok()) {
        return kLargeError;
      }
      cv::Vec2d ofs = computed->pos - sample.pos;
      error += ofs.ddot(ofs);
    }

    return error;
  }

  int getDims() const override { return 9; }

 private:
  absl::Span<const LaserCalibrationSample> samples_;
};

absl::StatusOr<std::pair<BoardLocationAndOrientation, double>>
ComputeBoardLocationWithError(
    absl::Span<const LaserCalibrationSample> samples) {
  cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
  cv::Ptr<cv::MinProblemSolver::Function> f(
      new ComputeBoardLocationProblem(samples));
  solver->setFunction(f);

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<double> dist(1, 5000);

  cv::Mat step(9, 1, CV_64FC1);
  for (int i = 0; i < 9; i++) {
    step.ptr<double>()[i] = dist(e2);
  }
  solver->setInitStep(step);

  cv::Mat x(9, 1, CV_64FC1);
  for (int i = 0; i < 9; i++) {
    x.ptr<double>()[i] = dist(e2);
  }
  double error = solver->minimize(x);

  ASSIGN_OR_RETURN(BoardLocationAndOrientation loc,
                   ComputeBoardLocationProblem::Unpack(x.ptr<double>()));
  return std::make_pair(loc, error);
}

absl::StatusOr<BoardLocationAndOrientation> ComputeBoardLocation(
    absl::Span<const LaserCalibrationSample> samples) {
  double error = std::numeric_limits<double>::max();
  std::optional<BoardLocationAndOrientation> loc;
  std::optional<absl::Status> status;
  for (int i = 0; i < kMaxMinimizationAttempts; i++) {
    auto res = ComputeBoardLocationWithError(samples);
    if (!res.ok()) {
      status.emplace(res.status());
      continue;
    }
    if (res->second < error) {
      error = res->second;
      loc.emplace(std::move(res->first));
    }
    if (error < 1) {
      return *loc;
    }
  }

  if (loc.has_value()) {
    return *loc;
  }
  if (status.has_value()) {
    return *status;
  }
  // Should be unreachable.
  return absl::InternalError("failed to find any board locations");
}

}  // namespace gobonline
