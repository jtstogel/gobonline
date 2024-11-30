#pragma once

/**
 * Macro to avoid repeating output matrix in OpenCV function calls
 * that are of the form `f(in, out, args...)`.
 *
 * Instead of writing:
 *   cv::Mat out;
 *   f(in, out, arg1, arg2);
 *
 * Write:
 *    CV_ASSIGN(cv::Mat, out, f, in, arg1, arg2);
 */
#define CV_ASSIGN(typ, out, f, in, ...) \
  typ out;                              \
  do {                                  \
    f(in, out, ##__VA_ARGS__);          \
  } while (false);

#define CV_REASSIGN_MAT(in, f, ...)                        \
  do {                                                     \
    cv::Mat tmp_variable_cv_macros_internal;               \
    f(in, tmp_variable_cv_macros_internal, ##__VA_ARGS__); \
    in = tmp_variable_cv_macros_internal;                  \
  } while (false);

#define CV_ASSIGN_MAT(out, f, in, ...) \
  CV_ASSIGN(cv::Mat, out, f, in, ##__VA_ARGS__)
