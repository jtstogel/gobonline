#include <vector>

#include "absl/status/statusor.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/objdetect/aruco_dictionary.hpp"

namespace gobonline {

struct Quadrangle2f {
  cv::Point2f top_left, top_right, bottom_left, bottom_right;
};

struct AruCoMarkerDesc {
  int length_millimeters = 40;
  cv::aruco::PredefinedDictionaryType aruco_dictionary =
      cv::aruco::DICT_4X4_100;
};

struct FindGobanOptions {
  // Markers in clockwise order from the top left corner,
  // ie [top left, top right, bottom right, bottom left].
  std::vector<int> aruco_marker_ids;

  // Descriptor for the AruCo markers in the image.
  AruCoMarkerDesc aruco_desc = AruCoMarkerDesc{};
};

/**
 * Returns the four corners of a goban's grid,
 * starting from the top left in clockwise order.
 */
absl::StatusOr<std::vector<cv::Point2f>> FindGoban(
    const cv::Mat& im, const FindGobanOptions& options);

}  // namespace gobonline
