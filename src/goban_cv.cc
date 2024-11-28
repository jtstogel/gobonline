#include "src/goban_cv.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "src/cv_macros.h"

namespace gobonline {
namespace {

using DetectedMarker = std::vector<cv::Point2f>;

/**
 * The approximate width of the Goban we'll be operating on.
 *
 * This number doesn't represent anything in the real world;
 * it's just what we'll resize the Goban to.
 */
constexpr int32_t kWorkingWidthPixels = 600;

/** Standard size of the empty space between two lines on a Goban. */
constexpr double kGobanLineSpacingMillimeters = 22;

/** Standard width of a line on a Goban. */
constexpr double kGobanLineWidthMillimeters = 1;

}  // namespace

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
  for (int i = 0; i < marker_ids.size(); i++) {
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
  return absl::StrCat("/home/jtstogel/github/jtstogel/gobonline/tmp/",
                      filename);
}

#define SAVE_DBG_IM(im) \
  cv::imwrite(IntermediatePath(absl::StrCat(#im, ".png")), im);

enum class Axis { kHorizontal = 0, kVertical = 1 };

template <typename T>
T Median(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
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
    size_t goban_width_px, const std::vector<DetectedMarker>& markers) {
  std::vector<cv::Point2f> inner_rect = AruCoGobanInnerRect(markers);

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

cv::Mat Normalized(const cv::Mat& m) {
  cv::Mat out;
  m.convertTo(out, CV_64F);
  out /= cv::sum(out);
  return out;
}

std::optional<Axis> LineAxis(const cv::Vec4f& line,
                             double slope_thresh = 0.05) {
  cv::Point2f p1(line[0], line[1]);
  cv::Point2f p2(line[2], line[3]);
  float dy = std::abs(p1.y - p2.y);
  float dx = std::abs(p1.x - p2.x);
  if (dy < slope_thresh * dx) {
    return Axis::kHorizontal;
  }
  if (dx < slope_thresh * dy) {
    return Axis::kVertical;
  }
  return std::nullopt;
}

std::vector<cv::Vec4f> FindGobanLines(const cv::Mat& im, double pixels_per_mm,
                                      Axis axis) {
  int spacing_width =
      MakeOdd(std::lround(kGobanLineSpacingMillimeters * pixels_per_mm));
  int line_width =
      MakeOdd(std::lround(kGobanLineWidthMillimeters * pixels_per_mm));

  cv::Size line_kernel_size = axis == Axis::kHorizontal
                                  ? cv::Size(line_width, spacing_width)
                                  : cv::Size(spacing_width, line_width);
  cv::Mat line_kernel =
      cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, line_kernel_size);

  // Approximate fraction of the Goban area filled in by lines.
  double line_proportion_area =
      2 * kGobanLineWidthMillimeters / kGobanLineSpacingMillimeters;
  cv::Ptr<cv::CLAHE> equalizer = cv::createCLAHE(
      /*clipLimit=*/255. * line_proportion_area,
      /*tileGridSize=*/cv::Size(1.5 * spacing_width, 1.5 * spacing_width));

  CV_ASSIGN_MAT(t, equalizer->apply, im);
  CV_APPLY_MAT(t, cv::dilate, /*kernel=*/line_kernel);
  CV_APPLY_MAT(t, cv::filter2D, /*ddepth=*/-1,
               /*kernel=*/Normalized(line_kernel));
  CV_APPLY_MAT(t, cv::Canny, /*threshold1=*/100, /*threshold2=*/200);
  CV_ASSIGN(std::vector<cv::Vec4f>, lines, cv::HoughLinesP, t, /*rho=*/1,
            /*theta=*/CV_PI / 180,
            /*threshold=*/80, /*minLineLength=*/3 * spacing_width);

  // Remove lines that aren't along the specified axis.
  std::vector<cv::Vec4f> axis_lines;
  axis_lines.reserve(lines.size());
  for (const cv::Vec4f& line : lines) {
    std::optional<Axis> line_axis = LineAxis(line);
    if (line_axis.has_value() && *line_axis == axis) {
      axis_lines.push_back(std::move(line));
    }
  }
  return lines;
}

std::optional<cv::Point2f> LineIntersection(const cv::Vec4f& l1,
                                            const cv::Vec4f& l2) {
  cv::Point2f a1(l1[0], l1[1]);
  cv::Point2f b1(l1[2], l1[3]);
  cv::Point2f a2(l2[0], l2[1]);
  cv::Point2f b2(l2[2], l2[3]);

  cv::Point2f x = a2 - a1;
  cv::Point2f d1 = b1 - a1;
  cv::Point2f d2 = b2 - a2;

  float cross = d1.x * d2.y - d1.y * d2.x;
  if (std::abs(cross) < 1e-7) {
    return std::nullopt;
  }

  double t1 = (x.x * d2.y - x.y * d2.x) / cross;
  if (t1 < 0 || 1 < t1) {
    return std::nullopt;
  }
  return a1 + d1 * t1;
}

/**
 * Finds the four corners of a Goban's grid in `im`,
 * given an overhead, aligned image of the goban.
 */
std::vector<cv::Point2f> FindGobanGridIntersections(const cv::Mat& im,
                                                    double pixels_per_mm) {
  std::vector<cv::Vec4f> vertical_lines =
      FindGobanLines(im, pixels_per_mm, Axis::kVertical);
  std::vector<cv::Vec4f> horizontal_lines =
      FindGobanLines(im, pixels_per_mm, Axis::kHorizontal);

  std::vector<cv::Point2f> intersections;
  for (const cv::Vec4f& vertical_line : vertical_lines) {
    for (const cv::Vec4f& horizontal_line : horizontal_lines) {
      std::optional<cv::Point2f> p =
          LineIntersection(vertical_line, horizontal_line);
      if (p.has_value()) {
        intersections.push_back(std::move(*p));
      }
    }
  }

  return intersections;
}

/**
 * Fits a size-by-size grid to the provided intersections.
 *
 * Returns the four corners of the grid in clockwise order starting from the
 * top-left corner.
 */
std::vector<cv::Point2f> FitGrid(const std::vector<cv::Point2f>& intersections,
                                 int size) {
  (void)size;
  return std::vector(intersections);
}

/**
 * Returns some calibration intrinsic to the physical Go board.
 */
absl::StatusOr<GobanFindingCalibration> ComputeGobanFindingCalibration(
    const cv::Mat& im, const FindGobanOptions& options) {
  absl::StatusOr<std::vector<DetectedMarker>> markers =
      FindGobanAruCoMarkers(im, options);
  if (!markers.ok()) return markers.status();

  std::vector<cv::Point2f> inner_rect = AruCoGobanInnerRect(*markers);
  GobanPerspectiveTransformParams xf_params =
      ComputeGobanGobanPerspectiveTransformParams(kWorkingWidthPixels,
                                                  *markers);

  std::vector<cv::Point2f> fixed_aruco_corner_locs = RectCorners(
      /*width=*/xf_params.aruco_box_size_px.width,
      /*height=*/xf_params.aruco_box_size_px.height, /*offset=*/0);
  cv::Mat xf = cv::getPerspectiveTransform(inner_rect, fixed_aruco_corner_locs);

  double pixels_per_mm =
      xf_params.aruco_size_px / options.aruco_markers.desc.length_millimeters;

  {
    SAVE_DBG_IM(im);
    CV_ASSIGN(cv::Mat, overhead_perspective, cv::warpPerspective, im, xf,
              xf_params.aruco_box_size_px);

    auto vert_lines =
        FindGobanLines(overhead_perspective, pixels_per_mm, Axis::kVertical);
    auto horz_lines =
        FindGobanLines(overhead_perspective, pixels_per_mm, Axis::kHorizontal);

    cv::Mat annotated_lines;
    im.copyTo(annotated_lines);

    std::vector<cv::Vec4f> lines;
    lines.insert(lines.end(), vert_lines.begin(), vert_lines.end());
    lines.insert(lines.end(), horz_lines.begin(), horz_lines.end());

    cv::Mat reverse_xf;
    cv::invert(xf, reverse_xf);
    for (auto& i : lines) {
      std::vector<cv::Point2f> points = {cv::Point2f(i[0], i[1]),
                                         cv::Point2f(i[2], i[3])};
      std::vector<cv::Point2f> points_in_original;
      cv::perspectiveTransform(points, points_in_original, reverse_xf);
      cv::line(annotated_lines, points_in_original[0], points_in_original[1],
               cv::Scalar(0, 0, 255), 1, 8);
    }

    SAVE_DBG_IM(annotated_lines);
  }

  {
    SAVE_DBG_IM(im);
    CV_ASSIGN(cv::Mat, overhead_perspective, cv::warpPerspective, im, xf,
              xf_params.aruco_box_size_px);

    cv::Mat annotated_intersections;
    im.copyTo(annotated_intersections);

    std::vector<cv::Point2f> intersections =
        FindGobanGridIntersections(overhead_perspective, pixels_per_mm);

    cv::Mat reverse_xf;
    cv::invert(xf, reverse_xf);
    std::vector<cv::Point2f> points_in_original;
    cv::perspectiveTransform(intersections, points_in_original, reverse_xf);

    for (auto& p : points_in_original) {
      cv::circle(annotated_intersections, p, 0, cv::Scalar(0, 0, 255));
    }

    SAVE_DBG_IM(annotated_intersections);
  }

  return GobanFindingCalibration{
      .options = options,
      .aruco_box_size_px = xf_params.aruco_box_size_px,
      .pixels_per_mm = pixels_per_mm,
  };
}

absl::StatusOr<std::vector<cv::Point2f>> FindGoban(
    const cv::Mat& im, const FindGobanOptions& options) {
  absl::StatusOr<std::vector<DetectedMarker>> markers =
      FindGobanAruCoMarkers(im, options);
  if (!markers.ok()) return markers.status();

  std::vector<cv::Point2f> goban_aruco_rect(4);
  for (int i = 0; i < 4; i++) {
    // AruCo corners are returned in clockwise order starting from the top left,
    // so offset the index by two in order to get the corner that is closest to
    // the Goban's grid corners.
    goban_aruco_rect.push_back(markers.value()[i][(i + 2) % 4]);
  }
  return goban_aruco_rect;
}

}  // namespace gobonline
