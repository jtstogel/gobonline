#include "src/goban_cv.h"

#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "src/util/status_test_utils.h"

namespace gobonline {

TEST(OpenCV, FindsGobanCorners) {
  cv::Mat im =
      cv::imread("src/testdata/empty_overhead.jpeg", cv::IMREAD_GRAYSCALE);
  FindGobanOptions find_options = {
      .aruco_markers = {.ids = {25, 14, 4, 7},
                        .desc = {.length_millimeters = 40}},
  };

  ASSERT_OK_AND_ASSIGN(GobanFindingCalibration calibration,
                       ComputeGobanFindingCalibration(im, find_options));

  ASSERT_OK_AND_ASSIGN(std::vector<cv::Point2f> aruco_corners,
                       FindGoban(im, find_options));

  cv::Size2f sz = calibration.aruco_box_size_px;
  cv::Mat xf = cv::getPerspectiveTransform(
      std::vector<cv::Point2f>(
          {{0, 0}, {sz.width, 0}, {sz.width, sz.height}, {0, sz.height}}),
      aruco_corners);

  std::vector<cv::Point2f> goban_corners;
  cv::perspectiveTransform(calibration.grid_corners, goban_corners, xf);

  std::cerr << "goban corners: ";
  for (const cv::Point2f& p : goban_corners) {
    std::cerr << "(" << p.x << "," << p.y << "), ";
  }
  std::cerr << std::endl;
}

}  // namespace gobonline
