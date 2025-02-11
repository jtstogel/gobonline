#include "src/laser_calibration_solver.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "opencv2/core/types.hpp"
#include "src/util/status_test_utils.h"

namespace gobonline {

using ::testing::DoubleNear;
using ::testing::Pointwise;

TEST(LaserCalibrationSolver, SimpleLaserSimulation) {
  // Pick a random location for the board.
  BoardLocationAndOrientation board = {
      .center_offset =
          cv::Vec3d{
              100,
              700,
              -900,
          },
      .x_axis_dir = cv::Vec3d(1, 0, 0),
      .y_axis_dir = cv::Vec3d(0, 1, 0),
  };

  ASSERT_OK_AND_ASSIGN(
      LaserCalibrationSample sample,
      SimulateLaserPath(
          MirrorAngles{
              .first_mirror_angle_radians = std::numbers::pi / 4,
              .second_mirror_angle_radians = std::numbers::pi / 8,
          },
          board));

  EXPECT_NEAR(sample.pos.x, -100, 0.1);
  EXPECT_NEAR(sample.pos.y, 900 - 700 + kMirrorDistanceMillimeters, 0.1);
}

std::vector<double> ToVec(const cv::Vec3d& v) {
  std::vector<double> r;
  r.push_back(v[0]);
  r.push_back(v[1]);
  r.push_back(v[2]);
  return r;
}

TEST(LaserCalibrationSolver, SimpleComputeBoardLocation) {
  // Pick a random location for the board.
  BoardLocationAndOrientation board_location = {
      .center_offset =
          cv::Vec3d{
              100,
              700,
              -900,
          },
      .x_axis_dir = cv::Vec3d(1, 0, 0),
      .y_axis_dir = cv::Vec3d(0, 1, 0),
  };
  std::vector<LaserCalibrationSample> samples;

  // Arbitrarily chosen ranges.
  double m1_lower = std::numbers::pi / 4 - std::numbers::pi / 16;
  double m1_upper = std::numbers::pi / 4 + std::numbers::pi / 16;
  double m2_lower = std::numbers::pi / 8 - std::numbers::pi / 20;
  double m2_upper = std::numbers::pi / 8 + std::numbers::pi / 20;
  for (float i = 0; i <= 1.; i += 0.1) {
    for (float j = 0; j < 1.; j += 0.1) {
      ASSERT_OK_AND_ASSIGN(LaserCalibrationSample s,
                           SimulateLaserPath(
                               MirrorAngles{
                                   .first_mirror_angle_radians =
                                       m1_lower + i * (m1_upper - m1_lower),
                                   .second_mirror_angle_radians =
                                       m2_lower + i * (m2_upper - m2_lower),
                               },
                               board_location));
      samples.push_back(s);
    }
  }

  ASSERT_OK_AND_ASSIGN(BoardLocationAndOrientation computed,
                       ComputeBoardLocation(samples));

  EXPECT_THAT(ToVec(computed.center_offset),
              Pointwise(DoubleNear(0.1), ToVec(board_location.center_offset)));
  EXPECT_THAT(ToVec(computed.x_axis_dir),
              Pointwise(DoubleNear(0.1), ToVec(board_location.x_axis_dir)));
  EXPECT_THAT(ToVec(computed.y_axis_dir),
              Pointwise(DoubleNear(0.1), ToVec(board_location.y_axis_dir)));
}

}  // namespace gobonline
