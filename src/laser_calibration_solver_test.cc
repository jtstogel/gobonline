#include "src/laser_calibration_solver.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>

#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/util/status_test_utils.h"

namespace gobonline {

using ::testing::DoubleNear;
using ::testing::Pointwise;

TEST(LaserCalibrationSolver, SimpleLaserSimulation) {
  BoardLocationAndOrientation board = {
      .origin_offset = Eigen::Vector3d{100, 700, -900},
      .x_axis = Eigen::Vector3d(1, 0, 0),
      .y_axis = Eigen::Vector3d(0, 1, 0),
  };

  ASSERT_OK_AND_ASSIGN(
      LaserPositionOnBoard p,
      ComputeLaserPositionOnBoard(
          MirrorAngles{
              .first_mirror_angle_radians = std::numbers::pi / 4,
              .second_mirror_angle_radians = std::numbers::pi / 8,
          },
          board));

  EXPECT_NEAR(p.position[0], -100, 0.1);
  EXPECT_NEAR(p.position[1], 900 - 700 + kMirrorDistanceMillimeters, 0.1);
}

TEST(LaserCalibrationSolver, SimpleComputeMirrorAngles) {
  BoardLocationAndOrientation board = {
      .origin_offset =
          Eigen::Vector3d{
              100,
              700,
              -900,
          },
      .x_axis = Eigen::Vector3d(1, 0, 0),
      .y_axis = Eigen::Vector3d(0, 1, 0),
  };
  LaserPositionOnBoard p = {
      .position = Eigen::Vector2d(0, 0),
  };

  absl::BitGen gen;
  ASSERT_OK_AND_ASSIGN(MirrorAngles angles,
                       ComputeLaserMirrorAngles(gen, board, p));

  EXPECT_NEAR(angles.first_mirror_angle_radians, 0.7433, 0.1);
  EXPECT_NEAR(angles.second_mirror_angle_radians, 0.3239, 0.1);
}

/** Tests whether `ComputeBoardLocation` accurately works on an example. */
void TestComputeBoardLocation(const BoardLocationAndOrientation& board) {
  LaserPositionOnBoard origin = {.position = {0, 0}};

  absl::BitGen gen;
  ASSERT_OK_AND_ASSIGN(MirrorAngles origin_angles,
                       ComputeLaserMirrorAngles(gen, board, origin));

  // This is approximately the step size available on a stepper motor.
  double step = std::numbers::pi / 100;

  std::vector<LaserCalibrationSample> samples;
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      MirrorAngles angles = {
          .first_mirror_angle_radians =
              origin_angles.first_mirror_angle_radians + i * step,
          .second_mirror_angle_radians =
              origin_angles.second_mirror_angle_radians + j * step,
      };
      ASSERT_OK_AND_ASSIGN(LaserPositionOnBoard p,
                           ComputeLaserPositionOnBoard(angles, board));
      samples.push_back({
          .position = p,
          .mirror_angles = angles,
      });
    }
  }

  std::cerr << "Board:\n"
            << "  origin: " << board.origin_offset.transpose() << "\n"  //
            << "  x_axis: " << board.x_axis.transpose() << "\n"         //
            << "  y_axis: " << board.y_axis.transpose() << "\n"         //
            << "  angles: " << origin_angles.first_mirror_angle_radians << ","
            << origin_angles.second_mirror_angle_radians << std::endl;

  ASSERT_OK_AND_ASSIGN(BoardLocationAndOrientation computed,
                       ComputeBoardLocation(gen, samples));

  EXPECT_THAT(computed.origin_offset.array(),
              Pointwise(DoubleNear(2.), board.origin_offset.array()));
  EXPECT_THAT(computed.x_axis.array(),
              Pointwise(DoubleNear(1e-2), board.x_axis.array()));
  EXPECT_THAT(computed.y_axis.array(),
              Pointwise(DoubleNear(1e-2), board.y_axis.array()));
}

TEST(LaserCalibrationSolver, StraightAheadBoardLocation) {
  TestComputeBoardLocation(BoardLocationAndOrientation{
      // The board is directly in front of the laser's position;
      // this is the easiest possible case.
      .origin_offset = Eigen::Vector3d{0, 700, kMirrorDistanceMillimeters},
      .x_axis = Eigen::Vector3d(1, 0, 0),
      .y_axis = Eigen::Vector3d(0, 0, 1),
  });
}

TEST(LaserCalibrationSolver, SimpleComputeBoardLocation) {
  TestComputeBoardLocation(BoardLocationAndOrientation{
      .origin_offset = Eigen::Vector3d{100, 700, -900},
      .x_axis = Eigen::Vector3d(1, 0, 0),
      .y_axis = Eigen::Vector3d(0, 1, 0),
  });
}

TEST(LaserCalibrationSolver, DifficultBoard1) {
  TestComputeBoardLocation(BoardLocationAndOrientation{
      .origin_offset = Eigen::Vector3d{-271.121, 763.74, -399.35},
      .x_axis = Eigen::Vector3d(0.528065, 0.0934828, 0.844043),
      .y_axis = Eigen::Vector3d(-0.848454, 0.0163266, 0.529017),
  });
}

TEST(LaserCalibrationSolver, DifficultBoard2) {
  TestComputeBoardLocation(BoardLocationAndOrientation{
      .origin_offset = Eigen::Vector3d{284.702, 858.317, 444.919},
      .x_axis = Eigen::Vector3d(0.584081, -0.0760211, -0.808128),
      .y_axis = Eigen::Vector3d(0.529687, -0.718702, 0.450444),
  });
}

std::optional<BoardLocationAndOrientation> RandomBoardLocation(
    absl::BitGen& gen) {
  auto unit = [&gen]() { return absl::Uniform<double>(gen, -1., 1.); };

  Eigen::Vector3d z_axis = {unit(), -std::fabs(unit()), unit()};
  z_axis.normalize();

  Eigen::Vector3d x_axis = {unit(), unit(), unit()};
  x_axis -= x_axis.dot(z_axis) * z_axis;
  x_axis.normalize();

  Eigen::Vector3d y_axis = z_axis.cross(x_axis);

  auto pos = [&gen]() {
    return absl::Uniform<double>(gen, -2 * kMmPerFoot, 2 * kMmPerFoot);
  };
  Eigen::Vector3d board_origin = {pos(), 4 * kMmPerFoot + pos(), pos()};

  // If the board isn't facing the camera enough, discard.
  double angle = std::acos(-board_origin.normalized().dot(z_axis));
  if (angle > std::numbers::pi / 4) {
    return std::nullopt;
  }

  return BoardLocationAndOrientation{
      .origin_offset = board_origin,
      .x_axis = x_axis,
      .y_axis = y_axis,
  };
}

BoardLocationAndOrientation RandomValidBoardLocation(absl::BitGen& gen) {
  while (true) {
    auto b = RandomBoardLocation(gen);
    if (b.has_value()) {
      return *b;
    }
  }
}

TEST(LaserCalibrationSolver, FuzzTest) {
  absl::BitGen gen;
  TestComputeBoardLocation(RandomValidBoardLocation(gen));
}

}  // namespace gobonline
