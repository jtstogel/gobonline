#include "src/goban_cv.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"

namespace gobonline {

absl::flat_hash_map<int, std::vector<cv::Point2f>> FindMarkers(
    const cv::Mat& im, const AruCoMarkerDesc& desc) {
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  std::vector<std::vector<cv::Point2f>> rejected_candidates;

  cv::aruco::DetectorParameters detector_params;
  detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(desc.aruco_dictionary);

  cv::aruco::ArucoDetector detector(dictionary, detector_params);
  detector.detectMarkers(im, marker_corners, marker_ids, rejected_candidates);

  absl::flat_hash_map<int, std::vector<cv::Point2f>> markers_by_id;
  for (int i = 0; i < marker_ids.size(); i++) {
    markers_by_id[marker_ids[i]] = marker_corners[i];
  }
  return markers_by_id;
}

absl::StatusOr<std::vector<cv::Point2f>> FindGoban(
    const cv::Mat& im, const FindGobanOptions& options) {
  if (options.aruco_marker_ids.size() != 4) {
    return absl::InvalidArgumentError(
        "FindGoban: aruco_marker_ids must contain exactly four distinct IDs");
  }

  auto markers = FindMarkers(im, options.aruco_desc);

  std::vector<cv::Point2f> goban_aruco_rect;
  std::vector<int> missing_ids;
  for (int i = 0; i < 4; i++) {
    int marker_id = options.aruco_marker_ids[i];
    auto it = markers.find(marker_id);
    if (it == markers.end()) {
      missing_ids.push_back(marker_id);
      continue;
    }

    // AruCo corners are returned in clockwise order starting from the top left,
    // so offset the index by two in order to get the corner that is closest to
    // the Goban's grid corners.
    goban_aruco_rect.push_back(it->second[(i + 2) % 4]);
  }

  if (!missing_ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find AruCo markers with ids ",
                     absl::StrJoin(missing_ids, ", ")));
  }

  return goban_aruco_rect;
}

}  // namespace gobonline
