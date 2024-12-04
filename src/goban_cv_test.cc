#include "src/goban_cv.h"

#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
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
  for (const auto& im : test_images.test_images()) {
    images.push_back(std::move(im));
  }

  return images;
}

std::string BaseName(absl::string_view s) {
  auto splitter = absl::StrSplit(s, '.');
  return std::string(*splitter.begin());
}

cv::Point2f TestDataPoint(const ::testdata::Point& p) {
  return cv::Point2f(p.column(), p.row());
}

std::string PointStr(const cv::Point2f& p) {
  return absl::StrCat("(", p.x, ",", p.y, ")");
}

std::string StringifyStones(const BoardState& state) {
  absl::flat_hash_map<std::pair<int32_t, int32_t>, Stone::Color> stones;
  for (const Stone& stone : state.stones()) {
    stones.insert_or_assign(
        std::make_pair(stone.position().row(), stone.position().column()),
        stone.color());
  }
  std::vector<std::string> rows;
  for (int32_t r = 0; r < 19; r++) {
    std::vector<std::string> row;
    for (int32_t c = 0; c < 19; c++) {
      auto it = stones.find(std::make_pair(r, c));
      if (it == stones.end()) {
        row.emplace_back(".");
      } else if (it->second == Stone::BLACK) {
        row.emplace_back("B");
      } else {
        row.emplace_back("W");
      }
    }
    rows.push_back(absl::StrJoin(row, " "));
  }
  return absl::StrJoin(rows, "\n");
}

TEST_P(GobanCVTest, FindsGobanCorners) {
  ::testdata::TestImage test_image = GetParam();
  cv::Mat im = cv::imread(absl::StrCat("src/testdata/", test_image.file_name()),
                          cv::IMREAD_GRAYSCALE);

  FindGobanOptions find_options = {
      .aruco_markers = {.ids = {25, 14, 4, 7},
                        .desc = {.length_millimeters = 40}},
      .grid_size = 19,
  };

  ASSERT_OK_AND_ASSIGN(GobanFindingCalibration calibration,
                       ComputeGobanFindingCalibration(
                           BaseName(test_image.file_name()), im, find_options));

  ASSERT_OK_AND_ASSIGN(
      BoardState state,
      ReadBoardState(BaseName(test_image.file_name()), im, calibration));

  std::vector<cv::Point2f> actual_corners;
  actual_corners.reserve(4);
  ASSERT_EQ(state.grid_corners().size(), 4);
  for (const Position2f& pos : state.grid_corners()) {
    actual_corners.emplace_back(pos.x(), pos.y());
  }

  std::vector<cv::Point2f> expected_corners = {
      TestDataPoint(test_image.goban_rectangle().top_left()),
      TestDataPoint(test_image.goban_rectangle().top_right()),
      TestDataPoint(test_image.goban_rectangle().bottom_right()),
      TestDataPoint(test_image.goban_rectangle().bottom_left()),
  };

  for (int i = 0; i < 4; i++) {
    double diff = cv::norm(expected_corners[i] - actual_corners[i]);
    ASSERT_LE(diff, 10) << absl::StrCat("corner[", i, "] has mismatch: got ",
                                        PointStr(actual_corners[i]), ", want ",
                                        PointStr(expected_corners[i]));
  }
}

TEST_P(GobanCVTest, FindsStones) {
  ::testdata::TestImage test_image = GetParam();
  cv::Mat im = cv::imread(absl::StrCat("src/testdata/", test_image.file_name()),
                          cv::IMREAD_GRAYSCALE);

  FindGobanOptions find_options = {
      .aruco_markers = {.ids = {25, 14, 4, 7},
                        .desc = {.length_millimeters = 40}},
      .grid_size = 19,
  };

  ASSERT_OK_AND_ASSIGN(GobanFindingCalibration calibration,
                       ComputeGobanFindingCalibration(
                           BaseName(test_image.file_name()), im, find_options));
  ASSERT_OK_AND_ASSIGN(
      BoardState state,
      ReadBoardState(BaseName(test_image.file_name()), im, calibration));

  std::string actual = StringifyStones(state);
  std::string expected = absl::StrJoin(test_image.stringified_stones(), "\n");

  EXPECT_EQ(expected, actual);
}

INSTANTIATE_TEST_SUITE_P(
    GobanCVTests, GobanCVTest, ::testing::ValuesIn(LoadTestImages()),
    [](const testing::TestParamInfo<::testdata::TestImage>& info) {
      return BaseName(info.param.file_name());
    });

}  // namespace gobonline
