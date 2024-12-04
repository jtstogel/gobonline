#include "gtest/gtest.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"

TEST(OpenCV, ImageRead) {
  cv::Mat img = cv::imread("src/testdata/empty_angled.jpeg", cv::IMREAD_COLOR);
  cv::Size size = img.size();

  EXPECT_EQ(size.width, 480);
  EXPECT_EQ(size.height, 640);
}
