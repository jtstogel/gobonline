#include <vector>

#include "absl/status/statusor.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/objdetect/aruco_dictionary.hpp"

namespace gobonline {

struct AruCoMarkerDesc {
  double length_millimeters = 40;
  cv::aruco::PredefinedDictionaryType aruco_dictionary =
      cv::aruco::DICT_4X4_100;
};

struct AruCoMarkers {
  // Marker IDs in clockwise order from the top left corner.
  std::vector<int> ids;

  // Descriptor for the AruCo markers in the image.
  AruCoMarkerDesc desc = AruCoMarkerDesc{};
};

struct FindGobanOptions {
  AruCoMarkers aruco_markers;
};

struct GobanFindingCalibration {
  FindGobanOptions options;

  // The size of the box outlined by the inner corners
  // of the four AruCo markers affixed to the Goban.
  //
  // All measurements below will use pixels.
  // This box anchors those measurements in a real-world, constant distance.
  cv::Size aruco_box_size_px;

  // Locations of the four corners of the grid.
  // (0, 0) is the location of the top left AruCo corner.
  std::vector<cv::Point2f> grid_corners;

  // Number of pixels per millimeter.
  double pixels_per_mm;
};

/**
 * Returns some calibration intrinsic to the physical Go board.
 *
 * The Goban should be mostly empty, as this function will use
 * the grid on the Goban for calibration.
 */
absl::StatusOr<GobanFindingCalibration> ComputeGobanFindingCalibration(
    const cv::Mat& im, const FindGobanOptions& options);

/**
 * Returns the four corners of a goban's grid,
 * starting from the top left in clockwise order.
 */
absl::StatusOr<std::vector<cv::Point2f>> FindGoban(
    const cv::Mat& im, const FindGobanOptions& options);

}  // namespace gobonline
