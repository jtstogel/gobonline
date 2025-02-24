#include "src/goban_cv.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "src/board_state.pb.h"
#include "src/cv_macros.h"
#include "src/util/simanneal/simanneal.h"
#include "src/util/status.h"
#include "src/util/union_find.h"

namespace gobonline {
namespace {

using DetectedMarker = std::vector<cv::Point2f>;

/** Approximate size of the empty space between two lines on a Goban. */
constexpr double kGobanLineSpacingMillimeters = 22;

constexpr int64_t kMinWorkingWidthPx = 400;
constexpr int64_t kMaxWorkingWidthPx = 800;
constexpr uint8_t kStoneColorThresh = 127;

absl::flat_hash_map<int, DetectedMarker> FindAruCoMarkers(
    const cv::Mat& im, const AruCoMarkerDesc& desc) {
  std::vector<int> marker_ids;
  std::vector<DetectedMarker> marker_corners;
  std::vector<DetectedMarker> rejected_candidates;

  cv::aruco::DetectorParameters detector_params;
  detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(desc.aruco_dictionary);

  cv::aruco::ArucoDetector detector(dictionary, detector_params);
  detector.detectMarkers(im, marker_corners, marker_ids, rejected_candidates);

  absl::flat_hash_map<int, DetectedMarker> markers_by_id;
  for (size_t i = 0; i < marker_ids.size(); i++) {
    markers_by_id[marker_ids[i]] = marker_corners[i];
  }
  return markers_by_id;
}

absl::StatusOr<std::vector<DetectedMarker>> FindGobanAruCoMarkers(
    const cv::Mat& im, const FindGobanOptions& options) {
  if (options.aruco_markers.ids.size() != 4) {
    return absl::InvalidArgumentError(
        "FindGoban: aruco_marker_ids must contain exactly four distinct IDs");
  }

  auto markers = FindAruCoMarkers(im, options.aruco_markers.desc);

  std::vector<DetectedMarker> goban_aruco_markers;
  std::vector<int> missing_ids;
  for (int i = 0; i < 4; i++) {
    int marker_id = options.aruco_markers.ids[i];
    auto it = markers.find(marker_id);
    if (it == markers.end()) {
      missing_ids.push_back(marker_id);
      continue;
    }
    goban_aruco_markers.push_back(std::move(it->second));
  }

  if (!missing_ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find AruCo markers with ids ",
                     absl::StrJoin(missing_ids, ", ")));
  }

  return goban_aruco_markers;
}

std::vector<cv::Point2f> RectCorners(float width, float height, float offset) {
  return {
      cv::Point2f(offset, offset),
      cv::Point2f(offset + width, offset),
      cv::Point2f(offset + width, offset + height),
      cv::Point2f(offset, offset + height),
  };
}

std::string IntermediatePath(absl::string_view filename) {
  std::filesystem::create_directory("/tmp/gobonline");
  std::filesystem::create_directory("/tmp/gobonline/debug");
  return absl::StrCat("/tmp/gobonline/debug/", filename);
}

#define SAVE_DBG_IM(im, ...)                                               \
  do {                                                                     \
    std::vector<absl::string_view> name_parts = {__VA_ARGS__};             \
    cv::imwrite(IntermediatePath(                                          \
                    absl::StrCat(absl::StrJoin(name_parts, "_"), ".png")), \
                im);                                                       \
  } while (false)

enum class Axis { kHorizontal = 0, kVertical = 1 };

template <typename T>
T Median(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double MedianRadius(const std::vector<cv::Vec3f>& circles) {
  std::vector<double> stone_radii;
  stone_radii.reserve(circles.size());
  for (const auto& circle : circles) {
    stone_radii.push_back(circle[2]);
  }
  if (stone_radii.empty()) {
    return 0;
  }
  return Median(stone_radii);
}

/**
 * Finds the length of an AruCo marker in pixels after applying the transform.
 * Markers are of a known real-world length, so this will help us pick
 * parameters for corner detection.
 */
float AruCoSideLength(const std::vector<DetectedMarker>& markers,
                      const cv::Mat& perspective_transform, Axis axis) {
  std::vector<float> aruco_side_lengths;
  for (const DetectedMarker& m : markers) {
    CV_ASSIGN(std::vector<cv::Point2f>, d, cv::perspectiveTransform, m,
              perspective_transform);
    for (int i = static_cast<int>(axis); i < 4; i += 2) {
      aruco_side_lengths.push_back(cv::norm(d[i] - d[(i + 1) % 4]));
    }
  }
  return Median(aruco_side_lengths);
}

std::vector<double> SideLengths(const std::vector<cv::Point2f>& rect) {
  std::vector<double> lengths;
  lengths.reserve(4);
  for (int i = 0; i < 4; i++) {
    lengths.push_back(cv::norm(rect[i] - rect[(i + 1) % 4]));
  }
  return lengths;
}

double MaxSideNorm(const std::vector<cv::Point2f>& rect) {
  std::vector<double> lengths = SideLengths(rect);
  return *std::ranges::max_element(lengths);
}

int MakeOdd(int x) { return x | 1; }

struct GobanPerspectiveTransformParams {
  // This defines an aspect ratio for the Goban.
  cv::Size aruco_box_size_px;
  double aruco_size_px;
};

std::vector<cv::Point2f> AruCoGobanInnerRect(
    const std::vector<DetectedMarker>& markers) {
  std::vector<cv::Point2f> inner_rect;
  inner_rect.reserve(4);
  for (int i = 0; i < 4; i++) {
    // Use the corners of each AruCo marker that
    // are closest to the Goban's grid.
    inner_rect.push_back(markers[i][(i + 2) % 4]);
  }
  return inner_rect;
}

GobanPerspectiveTransformParams ComputeGobanGobanPerspectiveTransformParams(
    const std::vector<DetectedMarker>& markers) {
  std::vector<cv::Point2f> inner_rect = AruCoGobanInnerRect(markers);

  // Use the maximum side length of the Goban for the dimensions
  // of the transformed image.
  int64_t goban_width_px = std::clamp(std::lround(MaxSideNorm(inner_rect)),
                                      kMinWorkingWidthPx, kMaxWorkingWidthPx);
  cv::Size(goban_width_px, goban_width_px);
  std::vector<cv::Point2f> aruco_square_corner_locs =
      RectCorners(goban_width_px, goban_width_px, /*offset=*/0);

  cv::Mat inner_rect_to_square_xf =
      cv::getPerspectiveTransform(inner_rect, aruco_square_corner_locs);

  float aruco_width_px =
      AruCoSideLength(markers, inner_rect_to_square_xf, Axis::kHorizontal);
  float aruco_height_px =
      AruCoSideLength(markers, inner_rect_to_square_xf, Axis::kVertical);

  cv::Mat aspect_ratio_fix_xf = cv::getPerspectiveTransform(
      RectCorners(aruco_width_px, aruco_height_px, /*offset=*/0),
      RectCorners(aruco_width_px, aruco_width_px, /*offset=*/0));
  cv::Mat xf = aspect_ratio_fix_xf * inner_rect_to_square_xf;

  CV_ASSIGN(std::vector<cv::Point2f>, rect, cv::perspectiveTransform,
            inner_rect, xf);

  return GobanPerspectiveTransformParams{
      .aruco_box_size_px =
          cv::Size((rect[1] - rect[0]).x, (rect[2] - rect[1]).y),
      .aruco_size_px = aruco_width_px,
  };
}

class Line {
 public:
  explicit Line(const cv::Vec4f& l) : p1_(l[0], l[1]), p2_(l[2], l[3]) {}

  /** Returns the axis this line falls on, if it falls along one. */
  std::optional<Axis> GetAxis(double slope_thresh = 0.05) const {
    float dy = std::abs(p1_.y - p2_.y);
    float dx = std::abs(p1_.x - p2_.x);
    if (dy < slope_thresh * dx) {
      return Axis::kHorizontal;
    }
    if (dx < slope_thresh * dy) {
      return Axis::kVertical;
    }
    return std::nullopt;
  }

  cv::Point2f Midpoint() const { return (p1_ + p2_) / 2; }

 private:
  cv::Point2f p1_;
  cv::Point2f p2_;
};

std::vector<float> JoinNearbyValues(std::vector<float>& values,
                                    float max_distance) {
  // Don't _really_ need a union find for this,
  // but it's a convenient abstraction.
  UnionFind uf(values.size());
  std::ranges::sort(values);
  for (size_t i = 0; i < values.size() - 1; i++) {
    if (std::abs(values[i] - values[i + 1]) <= max_distance) {
      uf.Union(i, i + 1);
    }
  }

  std::vector<float> joined_values;
  for (std::vector<size_t>& grouped_positions : uf.Sets()) {
    float avg_position = 0.;
    for (size_t idx : grouped_positions) {
      avg_position += values[idx];
    }
    avg_position /= grouped_positions.size();
    joined_values.push_back(avg_position);
  }

  return joined_values;
}

/**
 * Returns only the lines that lie along the specified axis.
 *
 * If Axis::kHorizontal, returns the y-coordinates of horizontal lines,
 * if Axis::kVertical, returns the x-coordinates of vertical lines.
 */
std::vector<float> PickAxisAlignedLines(absl::Span<const Line> lines,
                                        Axis axis) {
  std::vector<float> coordinates;
  coordinates.reserve(lines.size());
  for (const Line& l : lines) {
    std::optional<Axis> line_axis = l.GetAxis();
    if (line_axis.has_value() && *line_axis == axis) {
      cv::Point2f mid = l.Midpoint();
      coordinates.push_back(axis == Axis::kHorizontal ? mid.y : mid.x);
    }
  }
  return coordinates;
}

/**
 * Converts the given cv::Vec4f formatted lines to `Line` objects.
 */
std::vector<Line> VecsToLines(absl::Span<const cv::Vec4f> vector_lines) {
  std::vector<Line> lines;
  for (const cv::Vec4f& l : vector_lines) {
    lines.emplace_back(l);
  }
  return lines;
}

std::vector<float> FindGobanLines(absl::string_view debug_name,
                                  const cv::Mat& im, double pixels_per_mm,
                                  Axis axis) {
  int spacing_width =
      MakeOdd(std::lround(kGobanLineSpacingMillimeters * pixels_per_mm));

  int line_kernel_size = 2 * spacing_width;
  cv::Size line_kernel_size_2d = axis == Axis::kVertical
                                     ? cv::Size(1, line_kernel_size)
                                     : cv::Size(line_kernel_size, 1);
  cv::Mat line_morph_kernel = cv::getStructuringElement(
      cv::MorphShapes::MORPH_RECT, line_kernel_size_2d);
  cv::Mat line_filter_kernel = cv::Mat::zeros(line_kernel_size_2d, CV_64F);
  for (int i = 0; i < line_kernel_size; i++) {
    line_filter_kernel.ptr<double>()[i] = std::min(i, line_kernel_size - i);
  }
  line_filter_kernel /= cv::sum(line_filter_kernel);

  cv::Ptr<cv::CLAHE> equalizer = cv::createCLAHE(
      /*clipLimit=*/3.,
      /*tileGridSize=*/cv::Size(3 * spacing_width, 3 * spacing_width));

  CV_ASSIGN_MAT(equalized, equalizer->apply, im);
  CV_ASSIGN_MAT(dilated, cv::dilate, equalized, /*kernel=*/line_morph_kernel);
  CV_ASSIGN_MAT(filtered, cv::filter2D, dilated, /*ddepth=*/-1,
                /*kernel=*/line_filter_kernel);
  CV_ASSIGN_MAT(canny, cv::Canny, filtered, /*threshold1=*/100,
                /*threshold2=*/200);
  CV_ASSIGN(std::vector<cv::Vec4f>, hough_lines, cv::HoughLinesP, canny,
            /*rho=*/1,
            /*theta=*/CV_PI / 180,
            /*threshold=*/80, /*minLineLength=*/3 * spacing_width);

  {
    std::string direction =
        axis == Axis::kHorizontal ? "horizontal" : "vertical";
    SAVE_DBG_IM(equalized, debug_name, direction, "equalized");
    SAVE_DBG_IM(dilated, debug_name, direction, "dilate");
    SAVE_DBG_IM(filtered, debug_name, direction, "filter2D");
    SAVE_DBG_IM(canny, debug_name, direction, "canny");
  }

  std::vector<float> line_positions =
      PickAxisAlignedLines(VecsToLines(hough_lines), axis);

  // Canny edge detection may detect both sides of a line,
  // so we join detected lines that are very close together.
  std::vector<float> positions = JoinNearbyValues(
      line_positions, kGobanLineSpacingMillimeters * pixels_per_mm / 4);
  std::ranges::sort(positions);

  return positions;
}

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

/**
 * Finds the spacing of a 1D grid such that the sum of squared distances from
 * the points to the grid lines is minimized.
 */
class Fit1dGridProblem {
 public:
  static double FindSpacing(const std::vector<double>& points, int grid_size) {
    using SA = simanneal::SimulatedAnnealingOptimizer</*Dimensions=*/1>;

    auto r = std::ranges::minmax_element(points);
    SA optimizer(SA::Config{
        .bounds = {{{*r.min, *r.max - *r.min}}},
    });

    Fit1dGridProblem p(points, grid_size);

    absl::BitGen gen;
    absl::StatusOr<SA::Result> res = optimizer.Minimize(
        gen, [&p](const std::array<double, 1>& x) { return p.Error(x); });

    return res->x[0];
  }

  double Error(const std::array<double, 1>& x) const {
    double error = 0;
    for (double p : points_) {
      int64_t grid_line_idx =
          std::min<int64_t>(std::lround(p / x[0]), grid_size_);
      double offset = p - (grid_line_idx * x[0]);
      error += offset * offset;
    }
    return error;
  }

 private:
  Fit1dGridProblem(const std::vector<double>& points, int grid_size)
      : points_(points), grid_size_(grid_size) {}

  const std::vector<double>& points_;
  int grid_size_;
};

/**
 * Fits a 1D grid to the provided values.
 *
 * Returns the spacing between lines of the grid.
 */
double Fit1dGrid(const std::vector<float>& points, float center,
                 int grid_size) {
  std::vector<double> centered_points;
  centered_points.reserve(points.size());
  for (float p : points) {
    centered_points.push_back(std::abs(p - center));
  }

  return Fit1dGridProblem::FindSpacing(centered_points, grid_size / 2);
}

/**
 * Finds the four corners of a Goban's grid in `im`.
 *
 * The image of the goban must be:
 *   * Overhead - spacing between grid lines must be uniform.
 *   * Aligned  - lines are only horizontal and vertical.
 *   * Centered - Tengen is the center pixel.
 */
std::vector<cv::Point2f> FindGobanGrid(absl::string_view debug_name,
                                       const cv::Mat& im, double pixels_per_mm,
                                       int grid_size) {
  std::vector<float> vertical_line_xs =
      FindGobanLines(debug_name, im, pixels_per_mm, Axis::kVertical);
  std::vector<float> horizontal_line_ys =
      FindGobanLines(debug_name, im, pixels_per_mm, Axis::kHorizontal);

  cv::Point2f center = cv::Point2f(im.size().width, im.size().height) / 2;

  double vertical_line_spacing =
      Fit1dGrid(vertical_line_xs, center.x, grid_size);
  double horizontal_line_spacing =
      Fit1dGrid(horizontal_line_ys, center.y, grid_size);

  cv::Point2f dx = (grid_size - 1) / 2 * cv::Point2f(vertical_line_spacing, 0);
  cv::Point2f dy =
      (grid_size - 1) / 2 * cv::Point2f(0, horizontal_line_spacing);

  return {
      center - dx - dy,
      center + dx - dy,
      center + dx + dy,
      center - dx + dy,
  };
}

}  // namespace

/**
 * Returns some calibration intrinsic to the physical Go board.
 */
absl::StatusOr<GobanFindingCalibration> ComputeGobanFindingCalibration(
    absl::string_view debug_name, const cv::Mat& im,
    const FindGobanOptions& options) {
  SAVE_DBG_IM(im, debug_name, "original");

  ASSIGN_OR_RETURN(std::vector<DetectedMarker> markers,
                   FindGobanAruCoMarkers(im, options));

  GobanPerspectiveTransformParams xf_params =
      ComputeGobanGobanPerspectiveTransformParams(markers);

  std::vector<cv::Point2f> inner_rect = AruCoGobanInnerRect(markers);
  std::vector<cv::Point2f> fixed_aruco_corner_locs = RectCorners(
      /*width=*/xf_params.aruco_box_size_px.width,
      /*height=*/xf_params.aruco_box_size_px.height, /*offset=*/0);
  cv::Mat xf = cv::getPerspectiveTransform(inner_rect, fixed_aruco_corner_locs);

  double pixels_per_mm =
      xf_params.aruco_size_px / options.aruco_markers.desc.length_millimeters;

  CV_ASSIGN(cv::Mat, overhead_perspective, cv::warpPerspective, im, xf,
            xf_params.aruco_box_size_px);
  std::vector<cv::Point2f> grid_corners = FindGobanGrid(
      debug_name, overhead_perspective, pixels_per_mm, options.grid_size);

  return GobanFindingCalibration{
      .options = options,
      .aruco_box_size_px = xf_params.aruco_box_size_px,
      .grid_corners = grid_corners,
      .pixels_per_mm = pixels_per_mm,
  };
}

absl::StatusOr<BoardState> ReadBoardState(
    absl::string_view debug_name, const cv::Mat& im,
    const GobanFindingCalibration& calibration) {
  ASSIGN_OR_RETURN(std::vector<DetectedMarker> markers,
                   FindGobanAruCoMarkers(im, calibration.options));

  cv::Size2f sz = calibration.aruco_box_size_px;
  cv::Mat xf = cv::getPerspectiveTransform(
      AruCoGobanInnerRect(markers),
      std::vector<cv::Point2f>(
          {{0, 0}, {sz.width, 0}, {sz.width, sz.height}, {0, sz.height}}));

  CV_ASSIGN_MAT(overhead_board, cv::warpPerspective, im, xf,
                calibration.aruco_box_size_px);

  double spacing_width =
      calibration.pixels_per_mm * kGobanLineSpacingMillimeters;

  int morph_width = MakeOdd(std::lround(0.2 * spacing_width));
  cv::Size morph_size(morph_width, morph_width);
  cv::Mat morph_kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, morph_size);

  cv::medianBlur(overhead_board, overhead_board, morph_width);
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(overhead_board, circles, cv::HOUGH_GRADIENT_ALT, 2,
                   0.8 * spacing_width, 300, 0.75,
                   std::lround(0.5 * .83 * spacing_width),
                   std::lround(0.5 * 1.2 * spacing_width));

  {
    cv::Mat cirlces_im = overhead_board.clone();
    for (const auto& c : circles) {
      cv::Point2i center = cv::Point2i(c[0], c[1]);
      cv::circle(cirlces_im, center, 1, cv::Scalar(0, 100, 100), 3,
                 cv::LINE_AA);
      int radius = c[2];
      cv::circle(cirlces_im, center, radius, cv::Scalar(255, 0, 255), 3,
                 cv::LINE_AA);
      SAVE_DBG_IM(cirlces_im, debug_name, "with_circles");
    }
  }

  auto fit_to_grid = [calibration](double pt) {
    return std::clamp<int32_t>(
        std::lround(pt * (calibration.options.grid_size - 1)), 0,
        calibration.options.grid_size - 1);
  };

  cv::Point2f grid_origin = calibration.grid_corners[0];
  cv::Point2f grid_x = calibration.grid_corners[1] - grid_origin;
  cv::Point2f grid_y = calibration.grid_corners[3] - grid_origin;
  grid_x /= grid_x.ddot(grid_x);
  grid_y /= grid_y.ddot(grid_y);

  double stone_diameter = 2 * MedianRadius(circles);
  int filter_size = MakeOdd(std::lround(0.66 * stone_diameter));
  CV_ASSIGN_MAT(stone_color_im, cv::medianBlur, overhead_board, filter_size);
  SAVE_DBG_IM(stone_color_im, debug_name, "stone_blur");

  BoardState state;
  for (const auto& circle : circles) {
    cv::Point2f center(circle[0], circle[1]);
    Stone* stone = state.add_stones();
    stone->mutable_position()->set_column(
        fit_to_grid((center - grid_origin).ddot(grid_x)));
    stone->mutable_position()->set_row(
        fit_to_grid((center - grid_origin).ddot(grid_y)));

    uint8_t color = stone_color_im.at<uint8_t>(center.y, center.x);
    stone->set_color(color < kStoneColorThresh ? Stone::BLACK : Stone::WHITE);
  }

  CV_ASSIGN_MAT(inv_xf, cv::invert, xf);
  CV_ASSIGN(std::vector<cv::Point2f>, goban_corners, cv::perspectiveTransform,
            calibration.grid_corners, inv_xf);
  for (const cv::Point2f& corner : goban_corners) {
    Position2f* pos = state.add_grid_corners();
    pos->set_x(corner.x);
    pos->set_y(corner.y);
  }

  return state;
}

absl::StatusOr<LaserPosition> ReadLaserPosition(
    absl::string_view debug_name, const cv::Mat& im,
    const GobanFindingCalibration& calibration) {
  (void)debug_name;
  (void)im;
  (void)calibration;
  return absl::UnimplementedError("");
}

}  // namespace gobonline
