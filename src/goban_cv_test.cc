#include "src/goban_cv.h"

#include <filesystem>
#include <fstream>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "src/testdata/test_image.pb.h"
#include "src/util/status_test_utils.h"

namespace gobonline {

class GobanCVTest : public ::testing::TestWithParam<::testdata::TestImage> {};

std::vector<::testdata::TestImage> LoadTestImages() {
  std::ifstream t("src/testdata/test_images.textproto");
  std::stringstream ss;
  ss << t.rdbuf();

  ::testdata::TestImages test_images;
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &test_images)) {
    std::cerr << "Failed to parse test images!" << std::endl;
  }

  std::vector<::testdata::TestImage> images;
  for (auto& im : test_images.test_images()) {
    images.push_back(std::move(im));
  }

  return images;
}

cv::Point2f TestDataPoint(const ::testdata::Point& p) {
  return cv::Point2f(p.column(), p.row());
}

std::string PointStr(const cv::Point2f& p) {
  return absl::StrCat("(", p.x, ",", p.y, ")");
}

TEST_P(GobanCVTest, FindsGobanCorners) {
  ::testdata::TestImage test_image = GetParam();
  cv::Mat im = cv::imread(absl::StrCat("src/testdata/", test_image.file_name()),
                          cv::IMREAD_GRAYSCALE);

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

  std::vector<cv::Point2f> actual_corners = {
      TestDataPoint(test_image.goban_rectangle().top_left()),
      TestDataPoint(test_image.goban_rectangle().top_right()),
      TestDataPoint(test_image.goban_rectangle().bottom_right()),
      TestDataPoint(test_image.goban_rectangle().bottom_left()),
  };

  for (int i = 0; i < 4; i++) {
    double diff = cv::norm(goban_corners[i] - actual_corners[i]);
    ASSERT_LE(diff, 10) << absl::StrCat("corner[", i, "] has mismatch: got ",
                                        PointStr(goban_corners[i]), ", want ",
                                        PointStr(actual_corners[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    GobanCVTests, GobanCVTest, ::testing::ValuesIn(LoadTestImages()),
    [](const testing::TestParamInfo<::testdata::TestImage>& info) {
      std::string s(info.param.file_name());
      absl::StrReplaceAll({{".", ""}}, &s);
      return s;
    });

}  // namespace gobonline
