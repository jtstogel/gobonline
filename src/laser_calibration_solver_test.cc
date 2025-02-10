#include "src/laser_calibration_solver.h"

#include "gtest/gtest.h"
#include "opencv2/core/types.hpp"
#include "src/util/status_test_utils.h"

namespace gobonline {

TEST(GobanCVTest, CalibratesWhenNormal) {
  // Pick a random location for the board.
  BoardLocationAndOrientation board = {
      .center_offset =
          cv::Vec3d{
              150,
              700,
              -914,
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

  std::cerr << sample.pos.x << "," << sample.pos.y << std::endl;

  EXPECT_EQ(sample.pos.x, -150);
  EXPECT_EQ(sample.pos.y, 214);
}

}  // namespace gobonline
