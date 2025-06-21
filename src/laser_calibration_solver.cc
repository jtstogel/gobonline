#include "src/laser_calibration_solver.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/util/simanneal/simanneal.h"
#include "src/util/status.h"

namespace gobonline {

namespace {

using Eigen::Matrix3;
using Eigen::Vector2d;
using Eigen::Vector3;
using Eigen::Vector3d;
using Eigen::VectorXd;

template <size_t N>
VectorXd VecFromArray(const std::array<double, N>& array) {
  VectorXd v(array.size());
  for (size_t i = 0; i < array.size(); i++) {
    v(i) = array[i];
  }
  return v;
}

struct VectorLine {
  Vector3d direction;
  Vector3d origin;
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
      .origin = laser_offset,
  };
}

Vector3d LinePlaneIntesection(const Vector3d& line_direction,
                              const Vector3d& point_on_line,
                              const Vector3d& plane_normal,
                              const Vector3d& point_on_plane) {
  double td = line_direction.dot(plane_normal);

  static constexpr double kMinCorrelation = 1e-8;
  if (std::fabs(td) < kMinCorrelation) {
    td = (std::signbit(td) ? -1. : 1.) * kMinCorrelation;
  }

  double t = (point_on_plane - point_on_line).dot(plane_normal) / td;
  return point_on_line + t * line_direction;
}

template <typename T, int N, int M>
Eigen::Matrix<T, N, M> ConvertMatrixT(const Eigen::Matrix<double, N, M>& m) {
  Eigen::Matrix<T, N, M> result;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      result(i, j) = T(m(i, j));
    }
  }
  return result;
}

template <typename T>
Vector3<T> ConvertVector3(const Vector3<double>& m) {
  return ConvertMatrixT<T, 3, 1>(m);
}

template <typename T>
Matrix3<T> ConvertMatrix3(const Matrix3<double>& m) {
  return ConvertMatrixT<T, 3, 3>(m);
}

class BoardLocationProblem {
 private:
  template <typename T>
  struct Sample {
    Vector3<T> laser_direction;
    Vector3<T> laser_origin;

    // The laser interesection in board space (the z-component is set to 0).
    Vector3<T> laser_intersection_in_board_space;
  };

 public:
  explicit BoardLocationProblem(
      absl::Span<const LaserCalibrationSample> samples) {
    for (const LaserCalibrationSample& s : samples) {
      VectorLine laser = LaserLine(s.mirror_angles);
      Vector3d pos(s.position.position[0], s.position.position[1], 0);
      samples_.push_back(Sample<double>{
          .laser_direction = laser.direction,
          .laser_origin = laser.origin,
          .laser_intersection_in_board_space = pos,
      });
    }
  }

  template <typename T>
  T Error(const Eigen::VectorX<T>& x) {
    return Residuals<T>(x).squaredNorm() / samples_.size();
  }

  template <typename T>
  bool operator()(const T* const* x, T* residuals) const {
    Eigen::VectorX<T> x_vec(3);
    for (int i = 0; i < 3; i++) {
      x_vec[i] = x[0][i];
    }
    Eigen::VectorX<T> r = Residuals(x_vec);
    for (int i = 0; i < r.size(); i++) {
      residuals[i] = r(i);
    }
    return residuals;
  }

  template <typename T>
  Eigen::VectorX<T> Residuals(const Eigen::VectorX<T>& x) const {
    Vector3<T> origin;
    Eigen::Matrix3<T> rotation;
    return Residuals(x, origin, rotation);
  }

  LaserGalvoParameterization Unpack(const VectorXd& x) const {
    Eigen::Vector3d origin;
    Eigen::Matrix3d rotation;
    (void)Residuals(x, origin, rotation);
    return {
        .origin_offset = origin,
        .x_axis = rotation * Vector3d::UnitX(),
        .y_axis = rotation * Vector3d::UnitY(),
    };
  }

 private:
  template <typename T>
  Eigen::VectorX<T> Residuals(const Eigen::VectorX<T>& x, Vector3<T>& origin,
                              Matrix3<T>& rotation) const {
    rotation = Eigen::AngleAxis<T>(x(2), Vector3<T>::UnitZ()) *
               Eigen::AngleAxis<T>(x(1), Vector3<T>::UnitY()) *
               Eigen::AngleAxis<T>(x(0), Vector3<T>::UnitX());
    Vector3<T> normal = rotation * Vector3<T>::UnitZ();

    // The origin of the board can be directly solved for.
    std::vector<Matrix3<T>> projections;
    {
      Vector3<T> projected_origin = Vector3<T>::Zero();
      Matrix3<T> projections_sum = Eigen::Matrix3<T>::Zero();
      for (const Sample<double>& sample : samples_) {
        Sample<T> s = CastSample<T>(sample);

        Matrix3<T> projection =
            Matrix3<T>::Identity() - (s.laser_direction * normal.transpose() /
                                      s.laser_direction.dot(normal));
        projections.push_back(projection);
        projections_sum += projection.transpose() * projection;

        projected_origin += projection.transpose() *
                            (projection * s.laser_origin -
                             rotation * s.laser_intersection_in_board_space);
      }
      origin = projections_sum.inverse() * projected_origin;
    }

    if (origin[1] < 0.) {
      rotation = T(-1) * rotation;
      normal = T(-1) * normal;
      origin = T(-1) * origin;
    }

    Eigen::VectorX<T> residuals(3 * samples_.size());
    for (int i = 0; i < samples_.size(); i++) {
      Sample<T> s = CastSample<T>(samples_[i]);
      Vector3<T> residual = (projections[i] * (origin - s.laser_origin)) +
                            (rotation * s.laser_intersection_in_board_space);
      for (int j = 0; j < 3; j++) {
        residuals((3 * i) + j) = residual(j);
      }
    }
    return residuals;
  }

  template <typename T>
  Sample<T> CastSample(const Sample<double>& s) const {
    return Sample<T>{
        .laser_direction = ConvertVector3<T>(s.laser_direction),
        .laser_origin = ConvertVector3<T>(s.laser_origin),
        .laser_intersection_in_board_space =
            ConvertVector3<T>(s.laser_intersection_in_board_space),
    };
  }

  std::vector<Sample<double>> samples_;
};

Vector3d AverageLaserDirection(
    absl::Span<const LaserCalibrationSample> samples) {
  Vector3d avg_laser_direction = Vector3d::Zero();
  for (const LaserCalibrationSample& s : samples) {
    avg_laser_direction += LaserLine(s.mirror_angles).direction;
  }
  return avg_laser_direction.normalized();
}

class BoardLocationSolver {
 private:
  static constexpr size_t kDims = 3;

 public:
  static absl::StatusOr<LaserGalvoParameterization> SolveSimulatedAnnealing(
      absl::BitGenRef gen, absl::Span<const LaserCalibrationSample> samples) {
    BoardLocationProblem problem(samples);

    using Optimizer = simanneal::SimulatedAnnealingOptimizer<kDims>;
    Optimizer optimizer(Optimizer::Config{
        .bounds = kBounds,
        .initial_temperature = 500000,
        .max_iterations = 100000,
    });

    ASSIGN_OR_RETURN(
        Optimizer::Result result,
        optimizer.Minimize(gen,
                           [p = &problem](const std::array<double, kDims>& x) {
                             return p->Error(VecFromArray(x));
                           }));

    return problem.Unpack(VecFromArray(result.x));
  }

  static absl::StatusOr<LaserGalvoParameterization> SolveCeres(
      absl::Span<const LaserCalibrationSample> samples) {
    struct SolutionWithError {
      LaserGalvoParameterization board;
      double error;
    };
    BoardLocationProblem board_problem(samples);

    /**
     * Populate initial_guesses with Euler-angle representations
     * of the board's orientation. Our intial guess is a board that
     * is directly facing the average laser direction, with some
     * rotation about the board's normal.
     *
     * The function we're optimizing is fairly non-linear,
     * but one of these solutions should be close enough
     * to any physical situation to where most iteratives solvers
     * could perform well here.
     */
    constexpr int kGridSize = 16;
    std::array<std::array<double, 3>, kGridSize> initial_guesses;
    {
      Vector3d board_normal = -AverageLaserDirection(samples);
      Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(
          Eigen::Vector3d::UnitZ(), board_normal);
      for (int i = 0; i < kGridSize; i++) {
        double theta = 2 * std::numbers::pi * static_cast<double>(i) /
                       static_cast<double>(kGridSize);
        Eigen::AngleAxisd rotation_about_normal(theta, board_normal);
        Eigen::Matrix3d r =
            rotation_about_normal.toRotationMatrix() * q.toRotationMatrix();
        Vector3d x = r.eulerAngles(2, 1, 0);
        for (int j = 0; j < 3; j++) {
          initial_guesses[i][2 - j] = x[j];
        }
      }
    }

    SolutionWithError best = {.error = std::numeric_limits<double>::max()};
    for (std::array<double, 3>& x : initial_guesses) {
      auto* cost_fun =
          new ceres::DynamicAutoDiffCostFunction<BoardLocationProblem>(
              new BoardLocationProblem(board_problem));
      cost_fun->AddParameterBlock(3);
      cost_fun->SetNumResiduals(3 * samples.size());

      ceres::Problem problem;
      problem.AddParameterBlock(x.data(), 3);
      problem.AddResidualBlock(cost_fun, new ceres::HuberLoss(1.), x.data());

      ceres::Solver::Options options;
      options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
      options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
      options.max_num_iterations = 1000;
      options.function_tolerance = 1e-10;
      options.parameter_tolerance = 1e-10;
      options.gradient_tolerance = 1e-10;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      if (summary.IsSolutionUsable() && summary.final_cost < best.error) {
        best = SolutionWithError{
            .board = board_problem.Unpack(VecFromArray(x)),
            .error = summary.final_cost,
        };
      }
    }

    return best.board;
  }

 private:
  static constexpr std::array<std::array<double, 2>, kDims> kBounds = {{
      {0., 2 * std::numbers::pi},
      {0., 2 * std::numbers::pi},
      {0., 2 * std::numbers::pi},
  }};
};  // namespace

absl::Status CheckMirrorAngleBounds(double angle_radians) {
  if (angle_radians <= 0 || angle_radians >= std::numbers::pi / 2) {
    return absl::InvalidArgumentError(
        "mirror angle must be between 0 and pi/2");
  }
  return absl::OkStatus();
}

class MirrorAnglesSolver {
 public:
  static absl::StatusOr<MirrorAngles> Solve(
      const LaserGalvoParameterization& board,
      const LaserPositionOnBoard& position) {
    constexpr double kPi = std::numbers::pi;
    constexpr double kEps = 1e-8;
    auto atan2 = [](double x, double y) {
      return std::fmod(kPi + std::atan2(x, y), kPi);
    };

    Vector3d q = board.origin_offset + board.x_axis * position.position[0] +
                 board.y_axis * position.position[1];

    constexpr double kH = kMirrorDistanceMillimeters;
    MirrorAngles m = {
        .first_mirror_angle_radians = .5 * std::acos(q(0) / q.norm()),
        .second_mirror_angle_radians = .5 * atan2(q(1), kH - q(2)),
    };

    int max_iterations = 10;
    while (--max_iterations) {
      MirrorAngles prev = m;

      m.second_mirror_angle_radians = .5 * atan2(q(1), kH - q(2));
      m.first_mirror_angle_radians =
          .5 * atan2(kH + (q(1) / std::sin(2. * m.second_mirror_angle_radians)),
                     q(0));

      if (std::fabs(prev.first_mirror_angle_radians -
                    m.first_mirror_angle_radians) < kEps &&
          std::fabs(prev.second_mirror_angle_radians -
                    m.second_mirror_angle_radians) < kEps) {
        return m;
      }
    }
    return absl::InvalidArgumentError(
        "Failed to find valid mirror angles within 100 iterations");
  }
};

};  // namespace

absl::StatusOr<LaserPositionOnBoard> ComputeLaserPositionOnBoard(
    const MirrorAngles& angles, const LaserGalvoParameterization& board) {
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.first_mirror_angle_radians));
  RETURN_IF_ERROR(CheckMirrorAngleBounds(angles.second_mirror_angle_radians));

  Vector3d normal = board.x_axis.cross(board.y_axis);
  VectorLine laser_line = LaserLine(angles);
  Vector3d intersection = LinePlaneIntesection(
      laser_line.direction, laser_line.origin, normal, board.origin_offset);

  Vector3d location_relative_to_board_origin =
      intersection - board.origin_offset;
  return LaserPositionOnBoard{
      .position =
          {
              location_relative_to_board_origin.dot(board.x_axis),
              location_relative_to_board_origin.dot(board.y_axis),
          },
  };
}

absl::StatusOr<MirrorAngles> ComputeLaserMirrorAngles(
    const LaserGalvoParameterization& board,
    const LaserPositionOnBoard& position) {
  return MirrorAnglesSolver::Solve(board, position);
}

absl::StatusOr<LaserGalvoParameterization> ComputeBoardLocation(
    absl::Span<const LaserCalibrationSample> samples) {
  return BoardLocationSolver::SolveCeres(samples);
}

}  // namespace gobonline
