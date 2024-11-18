#include "src/goban_cv.h"

#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"

namespace gobonline {

#define ASSERT_OK(s) ASSERT_TRUE((s).ok()) << (s);

TEST(OpenCV, FindsGobanCorners) {
  cv::Mat im =
      cv::imread("src/testdata/empty_angled.jpeg", cv::IMREAD_GRAYSCALE);
  FindGobanOptions find_options = {
      .aruco_marker_ids = {25, 14, 4, 7},
  };

  absl::StatusOr<std::vector<cv::Point2f>> corners =
      FindGoban(im, find_options);
  ASSERT_OK(corners.status());

  std::cout << "Corners ";
  for (const auto& c : *corners) {
    std::cout << "(" << c.x << "," << c.y << "), ";
  }
  std::cout << std::endl;
}

}  // namespace gobonline
