#include "src/laser_calibration_solver.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <numbers>
#include <optional>

#include "absl/random/random.h"
#include "gtest/gtest.h"
#include "src/util/status_test_utils.h"

namespace gobonline {
namespace {


/*
Real life measurements.

With a real stepper motor, there is an error term in the angles that, per motor,
tends to be cyclic every 4 steps. On a K step stepper motor (K is usually 200 or 400),
one step is D = 2pi/K radians.

In addition to fitting our board parameters,
we should probably also try to fit that error term.

There's also a mechanical error for the stepper motor,
but seeing as that's completely unique per step,
it seems difficult to fit.

The error term is as a function of fmod(theta, 4 * D),
and it can only affect theta by at most +-5% of a step, so 0.1 * D.

We also define Error(0) = Error(4*D) = 0.

The error function we're fitting is that we're deciding on four distances.
The sum of these distances should be 4*D, and each distance should
be limited to (D*.95,D*1.05)

When an angle falls in a subsection in one of those distances,
we will correct it by the error in that subsection.

This does not take into mechanical error.

[{{0.789325,0.494801}, {-72.0074,-154.999}},
 ,{{0.789325,0.510509}, {-76.0756,-132.217}},
 ,{{0.789325,0.526217}, {-78.5166,-114.317}},
 ,{{0.789325,0.541925}, {-82.5848,-91.5348}},
 ,{{0.789325,0.557633}, {-85.0257,-74.4483}},
 ,{{0.789325,0.573341}, {-89.0939,-51.6663}},
 ,{{0.789325,0.589049}, {-92.3485,-34.5798}},
 ,{{0.789325,0.604757}, {-97.2303,-11.7978}},
 ,{{0.789325,0.620465}, {-99.6713,4.47504}},
 ,{{0.789325,0.636173}, {-105.367,28.8843}},
 ,{{0.789325,0.65188}, {-106.18,45.9708}},
 ,{{0.789325,0.667588}, {-110.249,69.5665}},
 ,{{0.789325,0.683296}, {-113.503,85.0257}},
 ,{{0.805033,0.683296}, {-93.1621,86.653}},
 ,{{0.805033,0.667588}, {-89.9076,71.1938}},
 ,{{0.805033,0.65188}, {-85.8393,46.7845}},
 ,{{0.805033,0.636173}, {-85.0257,32.1389}},
 ,{{0.805033,0.620465}, {-80.1438,8.54325}},
 ,{{0.805033,0.604757}, {-77.7029,-7.72961}},
 ,{{0.805033,0.589049}, {-73.6347,-31.3253}},
 ,{{0.805033,0.573341}, {-70.3801,-45.9708}},
 ,{{0.805033,0.557633}, {-65.4983,-68.7528}},
 ,{{0.805033,0.541925}, {-63.871,-86.653}},
 ,{{0.805033,0.526217}, {-59.8028,-109.435}},
 ,{{0.805033,0.510509}, {-55.7346,-127.335}},
 ,{{0.805033,0.494801}, {-51.6663,-150.117}},
 ,{{0.820741,0.494801}, {-27.257,-143.608}},
 ,{{0.820741,0.510509}, {-31.3253,-121.64}},
 ,{{0.820741,0.526217}, {-35.3935,-105.367}},
 ,{{0.820741,0.541925}, {-43.5299,-84.2121}},
 ,{{0.820741,0.557633}, {-46.7845,-67.9392}},
 ,{{0.820741,0.573341}, {-50.039,-45.9708}},
 ,{{0.820741,0.589049}, {-54.1073,-28.8843}},
 ,{{0.820741,0.604757}, {-57.3618,-6.91597}},
 ,{{0.820741,0.620465}, {-60.6164,10.1705}},
 ,{{0.820741,0.636173}, {-63.871,33.7662}},
 ,{{0.820741,0.65188}, {-65.4983,49.2254}},
 ,{{0.820741,0.667588}, {-70.3801,71.1938}},
 ,{{0.820741,0.683296}, {-73.6347,89.0939}},
 ,{{0.836449,0.683296}, {-44.3435,91.5348}},
 ,{{0.836449,0.667588}, {-43.5299,76.0756}},
 ,{{0.836449,0.65188}, {-39.4617,52.48}},
 ,{{0.836449,0.636173}, {-39.4617,37.0208}},
 ,{{0.836449,0.620465}, {-36.2071,14.2388}},
 ,{{0.836449,0.604757}, {-33.7662,-0.406822}},
 ,{{0.836449,0.589049}, {-28.8843,-23.1888}},
 ,{{0.836449,0.573341}, {-27.257,-39.4617}},
 ,{{0.836449,0.557633}, {-22.3752,-62.2437}},
 ,{{0.836449,0.541925}, {-21.5615,-77.7029}},
 ,{{0.836449,0.526217}, {-15.0524,-100.485}},
 ,{{0.836449,0.510509}, {-12.6115,-115.13}},
 ,{{0.836449,0.494801}, {-7.72961,-137.913}},
 ,{{0.852157,0.494801}, {18.307,-132.217}},
 ,{{0.852157,0.510509}, {12.6115,-111.876}},
 ,{{0.852157,0.526217}, {10.1705,-96.4167}},
 ,{{0.852157,0.541925}, {5.28868,-74.4483}},
 ,{{0.852157,0.557633}, {2.03411,-58.1755}},
 ,{{0.852157,0.573341}, {-1.22046,-37.0208}},
 ,{{0.852157,0.589049}, {-4.47504,-22.3752}},
 ,{{0.852157,0.604757}, {-8.54325,0.406822}},
 ,{{0.852157,0.620465}, {-10.9842,16.6797}},
 ,{{0.852157,0.636173}, {-14.2388,38.648}},
 ,{{0.852157,0.65188}, {-16.6797,54.1073}},
 ,{{0.852157,0.667588}, {-21.5615,76.8893}},
 ,{{0.852157,0.683296}, {-22.3752,92.3485}},
 ,{{0.867865,0.683296}, {-1.22046,93.1621}},
 ,{{0.867865,0.667588}, {-0.406822,79.3302}},
 ,{{0.867865,0.65188}, {3.66139,57.3618}},
 ,{{0.867865,0.636173}, {5.28868,41.089}},
 ,{{0.867865,0.620465}, {8.54325,21.5615}},
 ,{{0.867865,0.604757}, {10.9842,4.47504}},
 ,{{0.867865,0.589049}, {15.0524,-16.6797}},
 ,{{0.867865,0.573341}, {19.1206,-32.1389}},
 ,{{0.867865,0.557633}, {19.9343,-53.2936}},
 ,{{0.867865,0.541925}, {24.0025,-68.7528}},
 ,{{0.867865,0.526217}, {28.8843,-91.5348}},
 ,{{0.867865,0.510509}, {32.1389,-106.994}},
 ,{{0.867865,0.494801}, {37.0208,-128.149}},
 ,{{0.883573,0.494801}, {60.6164,-122.453}},
 ,{{0.883573,0.510509}, {54.1073,-102.112}},
 ,{{0.883573,0.526217}, {50.8527,-86.653}},
 ,{{0.883573,0.541925}, {41.089,-66.3119}},
 ,{{0.883573,0.557633}, {40.2753,-50.8527}},
 ,{{0.883573,0.573341}, {35.3935,-30.5116}},
 ,{{0.883573,0.589049}, {32.1389,-15.866}},
 ,{{0.883573,0.604757}, {28.8843,5.28868}},
 ,{{0.883573,0.620465}, {26.4434,22.3752}},
 ,{{0.883573,0.636173}, {23.1888,41.9026}},
 ,{{0.883573,0.65188}, {19.9343,58.1755}},
 ,{{0.883573,0.667588}, {19.1206,79.3302}},
 ,{{0.883573,0.683296}, {15.0524,93.1621}},
 ,{{0.899281,0.683296}, {40.2753,95.6031}},
 ,{{0.899281,0.667588}, {41.089,82.5848}},
 ,{{0.899281,0.65188}, {44.3435,61.4301}},
 ,{{0.899281,0.636173}, {44.3435,45.9708}},
 ,{{0.899281,0.620465}, {47.5981,24.0025}},
 ,{{0.899281,0.604757}, {50.039,10.1705}},
 ,{{0.899281,0.589049}, {54.1073,-10.9842}},
 ,{{0.899281,0.573341}, {56.5482,-24.0025}},
 ,{{0.899281,0.557633}, {59.8028,-45.9708}},
 ,{{0.899281,0.541925}, {62.2437,-61.4301}},
 ,{{0.899281,0.526217}, {67.1256,-82.5848}},
 ,{{0.899281,0.510509}, {71.1938,-98.044}},
 ,{{0.899281,0.494801}, {75.262,-117.571}},
 ,{{0.914989,0.494801}, {98.8576,-113.503}},
 ,{{0.914989,0.510509}, {94.7894,-92.3485}},
 ,{{0.914989,0.526217}, {91.5348,-77.7029}},
 ,{{0.914989,0.541925}, {87.4666,-58.1755}},
 ,{{0.914989,0.557633}, {83.3984,-42.7163}},
 ,{{0.914989,0.573341}, {80.9575,-23.1888}},
 ,{{0.914989,0.589049}, {77.7029,-7.72961}},
 ,{{0.914989,0.604757}, {73.6347,11.7978}},
 ,{{0.914989,0.620465}, {72.0074,26.4434}},
 ,{{0.914989,0.636173}, {68.7528,45.9708}},
 ,{{0.914989,0.65188}, {66.3119,62.2437}},
 ,{{0.914989,0.667588}, {62.2437,83.3984}},
 ,{{0.914989,0.683296}, {62.2437,97.2303}},
 ,{{0.930697,0.683296}, {82.5848,98.044}},
 ,{{0.930697,0.667588}, {82.5848,85.0257}},
 ,{{0.930697,0.65188}, {85.0257,65.4983}},
 ,{{0.930697,0.636173}, {86.653,50.039}},
 ,{{0.930697,0.620465}, {89.9076,29.698}},
 ,{{0.930697,0.604757}, {92.3485,15.0524}},
 ,{{0.930697,0.589049}, {95.6031,-4.47504}},
 ,{{0.930697,0.573341}, {98.044,-19.1206}},
 ,{{0.930697,0.557633}, {102.926,-38.648}},
 ,{{0.930697,0.541925}, {103.739,-53.2936}},
 ,{{0.930697,0.526217}, {109.435,-73.6347}},
 ,{{0.930697,0.510509}, {112.69,-88.2803}},
 ,{{0.930697,0.494801}, {116.758,-107.808}},
 ,{{0.946405,0.494801}, {137.913,-103.739}},
 ,{{0.946405,0.510509}, {133.031,-84.2121}},
 ,{{0.946405,0.526217}, {129.776,-69.5665}},
 ,{{0.946405,0.541925}, {120.826,-50.8527}},
 ,{{0.946405,0.557633}, {118.385,-37.0208}},
 ,{{0.946405,0.573341}, {114.317,-16.6797}},
 ,{{0.946405,0.589049}, {112.69,-2.84775}},
 ,{{0.946405,0.604757}, {108.621,16.6797}},
 ,{{0.946405,0.620465}, {104.553,31.3253}},
 ,{{0.946405,0.636173}, {103.739,51.6663}},
 ,{{0.946405,0.65188}, {102.926,65.4983}},
 ,{{0.946405,0.667588}, {98.8576,85.0257}},
 ,{{0.946405,0.683296}, {97.2303,98.8576}},
 ,{{0.962113,0.683296}, {120.826,100.485}},
 ,{{0.962113,0.667588}, {124.081,87.4666}},
 ,{{0.962113,0.65188}, {125.708,69.5665}},
 ,{{0.962113,0.636173}, {125.708,53.2936}},
 ,{{0.962113,0.620465}, {126.521,35.3935}},
 ,{{0.962113,0.604757}, {129.776,22.3752}},
 ,{{0.962113,0.589049}, {133.844,0.406822}},
 ,{{0.962113,0.573341}, {135.472,-12.6115}},
 ,{{0.962113,0.557633}, {139.54,-32.1389}},
 ,{{0.962113,0.541925}, {142.794,-46.7845}},
 ,{{0.962113,0.526217}, {146.049,-65.4983}},
 ,{{0.962113,0.510509}, {149.304,-79.3302}},
 ,{{0.962113,0.494801}, {154.185,-98.8576}},
 ,{{0.977821,0.494801}, {175.34,-93.1621}},
 ,{{0.977821,0.510509}, {170.458,-76.0756}},
 ,{{0.977821,0.526217}, {167.204,-61.4301}},
 ,{{0.977821,0.541925}, {162.322,-42.7163}},
 ,{{0.977821,0.557633}, {159.067,-28.8843}},
 ,{{0.977821,0.573341}, {155.813,-10.1705}},
 ,{{0.977821,0.589049}, {153.372,2.03411}},
 ,{{0.977821,0.604757}, {149.304,23.1888}},
 ,{{0.977821,0.620465}, {146.863,37.0208}},
 ,{{0.977821,0.636173}, {146.049,54.9209}},
 ,{{0.977821,0.65188}, {141.981,69.5665}},
 ,{{0.977821,0.667588}, {141.167,88.2803}},
 ,{{0.977821,0.683296}, {139.54,101.299}},
]
*/

/** Tests whether `ComputeBoardLocation` accurately works on an example. */
void TestComputeBoardLocation(const LaserGalvoParameterization& board,
                              double noise = 0.) {
  LaserPositionOnBoard origin = {.position = {0, 0}};

  ASSERT_OK_AND_ASSIGN(MirrorAngles origin_angles,
                       ComputeLaserMirrorAngles(board, origin));

  // This is approximately the step size available on a stepper motor.
  double step = std::numbers::pi / 100;

  absl::BitGen gen;
  auto noise_dist = [&gen, noise]() {
    return absl::Gaussian<double>(gen, 0, noise);
  };
  std::vector<LaserCalibrationSample> samples;
  for (int i = -3; i <= 3; i++) {
    for (int j = -3; j <= 3; j++) {
      MirrorAngles angles = {
          .first_mirror_angle_radians =
              origin_angles.first_mirror_angle_radians + (i * step),
          .second_mirror_angle_radians =
              origin_angles.second_mirror_angle_radians + (j * step),
      };
      ASSERT_OK_AND_ASSIGN(LaserPositionOnBoard p,
                           ComputeLaserPositionOnBoard(angles, board));
      p.position += Eigen::Vector2d(noise_dist(), noise_dist());
      samples.push_back({
          .position = p,
          .mirror_angles = angles,
      });
    }
  }

  std::cerr << "Actual board:\n"
            << "  origin: " << board.origin_offset.transpose() << "\n"  //
            << "  x_axis: " << board.x_axis.transpose() << "\n"         //
            << "  y_axis: " << board.y_axis.transpose() << "\n"         //
            << "  angles: " << origin_angles.first_mirror_angle_radians << ","
            << origin_angles.second_mirror_angle_radians << std::endl;

  ASSERT_OK_AND_ASSIGN(LaserGalvoParameterization computed,
                       ComputeBoardLocation(samples));

  std::cerr << "Computed board:\n"
            << "  origin: " << computed.origin_offset.transpose() << "\n"  //
            << "  x_axis: " << computed.x_axis.transpose() << "\n"         //
            << "  y_axis: " << computed.y_axis.transpose() << std::endl;

  EXPECT_LT((board.origin_offset - computed.origin_offset).norm(),
            .1 + (4 * noise));
  EXPECT_GT(computed.x_axis.dot(board.x_axis), 1. - 1e-3);
  EXPECT_GT(computed.y_axis.dot(board.y_axis), 1. - 1e-3);
}

std::optional<LaserGalvoParameterization> RandomBoardLocation(
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
  Eigen::Vector3d board_origin = {pos(), (4 * kMmPerFoot) + pos(), pos()};

  // If the board isn't facing the camera enough, discard.
  double angle = std::acos(-board_origin.normalized().dot(z_axis));
  if (angle > std::numbers::pi / 3) {
    return std::nullopt;
  }

  return LaserGalvoParameterization{
      .origin_offset = board_origin,
      .x_axis = x_axis,
      .y_axis = y_axis,
  };
}

LaserGalvoParameterization RandomValidBoardLocation(absl::BitGen& gen) {
  while (true) {
    auto b = RandomBoardLocation(gen);
    if (b.has_value()) {
      return *b;
    }
  }
}

}  // namespace

TEST(LaserCalibrationSolver, SimpleLaserSimulation) {
  LaserGalvoParameterization board = {
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
  LaserGalvoParameterization board = {
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

  ASSERT_OK_AND_ASSIGN(MirrorAngles angles, ComputeLaserMirrorAngles(board, p));

  EXPECT_NEAR(angles.first_mirror_angle_radians, 0.7433, 1e-5);
  EXPECT_NEAR(angles.second_mirror_angle_radians, 0.3239, 1e-5);
}

TEST(LaserCalibrationSolver, StraightAheadBoardLocation) {
  TestComputeBoardLocation(LaserGalvoParameterization{
      // The board is directly in front of the laser's position;
      // this is the easiest possible case.
      .origin_offset = Eigen::Vector3d{0, 700, kMirrorDistanceMillimeters},
      .x_axis = Eigen::Vector3d(1, 0, 0),
      .y_axis = Eigen::Vector3d(0, 0, 1),
  });
}

TEST(LaserCalibrationSolver, SimpleComputeBoardLocation) {
  TestComputeBoardLocation(LaserGalvoParameterization{
      .origin_offset = Eigen::Vector3d{100, 700, -900},
      .x_axis = Eigen::Vector3d(1, 0, 0),
      .y_axis = Eigen::Vector3d(0, 1, 0),
  });
}

TEST(LaserCalibrationSolver, DifficultBoard1) {
  TestComputeBoardLocation(LaserGalvoParameterization{
      .origin_offset = Eigen::Vector3d{-271.121, 763.74, -399.35},
      .x_axis = Eigen::Vector3d(0.528065, 0.0934828, 0.844043),
      .y_axis = Eigen::Vector3d(-0.848454, 0.0163266, 0.529017),
  });
}

TEST(LaserCalibrationSolver, DifficultBoard2) {
  TestComputeBoardLocation(LaserGalvoParameterization{
      .origin_offset = Eigen::Vector3d{284.702, 858.317, 444.919},
      .x_axis = Eigen::Vector3d(0.584081, -0.0760211, -0.808128),
      .y_axis = Eigen::Vector3d(0.529687, -0.718702, 0.450444),
  });
}

TEST(LaserCalibrationSolver, FuzzTest) {
  absl::BitGen gen;
  TestComputeBoardLocation(RandomValidBoardLocation(gen));
}

}  // namespace gobonline
