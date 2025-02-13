#include "src/laser_calibration_solver.h"

#include <Eigen/Core>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/util/status_test_utils.h"

namespace gobonline {

using ::testing::DoubleNear;
using ::testing::Pointwise;

TEST(LaserCalibrationSolver, SimpleLaserSimulation) {
  BoardLocationAndOrientation board = {
      .center_offset =
          Eigen::Vector3d{
              100,
              700,
              -900,
          },
      .normal = Eigen::Vector3d(0, 0, 1),
      .x_axis = Eigen::Vector3d(1, 0, 0),
  };

  ASSERT_OK_AND_ASSIGN(
      LaserCalibrationSample sample,
      SimulateLaserPath(
          MirrorAngles{
              .first_mirror_angle_radians = std::numbers::pi / 4,
              .second_mirror_angle_radians = std::numbers::pi / 8,
          },
          board));

  EXPECT_NEAR(sample.pos[0], -100, 0.1);
  EXPECT_NEAR(sample.pos[1], 900 - 700 + kMirrorDistanceMillimeters, 0.1);
}

TEST(LaserCalibrationSolver, SimpleComputeBoardLocation) {
  BoardLocationAndOrientation board_location = {
      .center_offset =
          Eigen::Vector3d{
              100,
              700,
              -900,
          },
      .normal = Eigen::Vector3d(0, 0, 1),
      .x_axis = Eigen::Vector3d(1, 0, 0),
  };
  std::vector<LaserCalibrationSample> samples;

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<double> m1(
      std::numbers::pi / 4 - std::numbers::pi / 16,
      std::numbers::pi / 4 + std::numbers::pi / 16);
  std::uniform_real_distribution<double> m2(
      std::numbers::pi / 8 - std::numbers::pi / 20,
      std::numbers::pi / 8 + std::numbers::pi / 20);

  for (int i = 0; i <= 20; i++) {
    ASSERT_OK_AND_ASSIGN(LaserCalibrationSample s,
                         SimulateLaserPath(
                             MirrorAngles{
                                 .first_mirror_angle_radians = m1(e2),
                                 .second_mirror_angle_radians = m2(e2),
                             },
                             board_location));
    samples.push_back(s);
  }

  ASSERT_OK_AND_ASSIGN(BoardLocationAndOrientation computed,
                       ComputeBoardLocation(samples));

  EXPECT_THAT(computed.center_offset.array(),
              Pointwise(DoubleNear(5.), board_location.center_offset.array()));
  EXPECT_THAT(computed.x_axis.array(),
              Pointwise(DoubleNear(0.1), board_location.x_axis.array()));
  EXPECT_THAT(computed.normal.array(),
              Pointwise(DoubleNear(0.1), board_location.normal.array()));
}

TEST(LaserCalibrationSolver, PercentOk) {
  // Pick a random location for the board.
  BoardLocationAndOrientation board_location = {
      .center_offset =
          Eigen::Vector3d{
              0,
              1200,
              0,
          },
      .normal = Eigen::Vector3d(0, -1/std::sqrt(2), 1/std::sqrt(2)),
      .x_axis = Eigen::Vector3d(1, 0, 0),
  };
  std::vector<LaserCalibrationSample> samples;

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<double> angle(
      std::numbers::pi / 4 - std::numbers::pi / 32,
      std::numbers::pi / 4 + std::numbers::pi / 32);

  for (int i = 0; i <= 20; i++) {
    ASSERT_OK_AND_ASSIGN(LaserCalibrationSample s,
                         SimulateLaserPath(
                             MirrorAngles{
                                 .first_mirror_angle_radians = angle(e2),
                                 .second_mirror_angle_radians = angle(e2),
                             },
                             board_location));
    samples.push_back(s);
    std::cerr << "Sample: " << s.mirror_angles.first_mirror_angle_radians << " "
              << s.mirror_angles.second_mirror_angle_radians << " "
              << s.pos.transpose() << std::endl;
  }

  int successes = 0;
  int total = 0;
  for (int i = 0; i < 1000; i++) {
    total++;
    successes += ComputeBoardLocation(samples).ok() ? 1 : 0;
  }
  std::cerr << "success count: " << successes << " / " << total << std::endl;
  EXPECT_EQ(successes, total);
}

}  // namespace gobonline
