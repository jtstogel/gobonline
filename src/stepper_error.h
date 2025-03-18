#pragma once

#include <Eigen/Core>
#include <cassert>
#include <iostream>
#include <numbers>

namespace gobonline {

/** Number of steps in a 360 degree rotation of the stepper motor. */
constexpr double kMotorStepsPerRevolution = 400;

/** Number of steps in a 360 degree rotation of the stepper motor. */
constexpr double kRadiansPerStep =
    2 * std::numbers::pi / kMotorStepsPerRevolution;

/** Number of steps in the periodic error of the stepper motor. */
constexpr int kStepsPerElectricalErrorPeriod = 4;

/** Radians in an error period. */
constexpr double kRadiansPerErrorPeriod =
    kRadiansPerStep * kStepsPerElectricalErrorPeriod;

// Two phase stepper motors intrinsically have two different
// types of error: mechanical and eletrical.
//
// Mechanical error in a step is unique per step location
// and so is impossible to model without completely characterizing
// the motor. Electrical error for two-phase steppers repeats
// every 4 steps. The error offsets a step's location by
// at most 5% in either direction of the stepper.
struct StepperError {
  // The errors in each step in a 4-step period.
  //
  // If the below is a perfectly even step:
  // |-----|-----|-----|-----|
  //
  // Then the errors {-1, 1, 1} would look like:
  // |----|-------|---|------|
  Eigen::Vector<double, kStepsPerElectricalErrorPeriod - 1> error_per_step;

  // A value in [0,1) that describes the offset the angle
  // within a 4 step period.
  //
  // It may be the case that the zero angle does not align
  // exactly with a step (if using microstepping), so this
  // number describes the amount within an error period that
  // we need to offset before we can describe the cycle.
  double period_offset;
};

template <size_t N, double Min = 0., double Max = 1.>
class DiscreteUniformIntervalsError {
 public:
  explicit DiscreteUniformIntervalsError(
      const std::array<double, N - 1>& errors) {
    errors_ = ComputeCumulativeOffsets(errors);

    std::array<double, N - 1> zero_errors = {0};
    even_ = ComputeCumulativeOffsets(zero_errors);
  }

  double AddError(double t) const { return Convert(t, even_, errors_); }

  double RemoveError(double t) const { return Convert(t, errors_, even_); }

 private:
  using CumulativeOffsets = std::array<double, N + 1>;
  static constexpr double kTotalSize = Max - Min;
  static constexpr double kIntervalSize = kTotalSize / static_cast<double>(N);

  CumulativeOffsets ComputeCumulativeOffsets(
      const std::array<double, N - 1>& errors) {
    CumulativeOffsets offsets;
    offsets[0] = Min;
    offsets[N] = Max;
    for (int i = 1; i < N; i++) {
      offsets[i] = Min + i * kIntervalSize + errors[i - 1];
    }
    return offsets;
  }

  CumulativeOffsets even_;
  CumulativeOffsets errors_;

  double Convert(double angle, const CumulativeOffsets& source,
                 const CumulativeOffsets& dest) const {
    double angle_in_interval =
        std::fmod(std::fmod(angle - Min, kTotalSize) + kTotalSize, kTotalSize);
    double xfm_angle_in_interval =
        ConvertWithinInterval(angle_in_interval, source, dest);
    return angle + (xfm_angle_in_interval - angle_in_interval);
  }

  double ConvertWithinInterval(double angle, const CumulativeOffsets& source,
                               const CumulativeOffsets& dest) const {
    for (int i = 0; i < source.size() - 1; i++) {
      if (source[i] <= angle && angle < source[i + 1]) {
        double t = (angle - source[i]) / (source[i + 1] - source[i]);
        std::cerr << "t:" << t << std::endl;
        return dest[i] + (t * (dest[i + 1] - dest[i]));
      }
    }
    return Max;
  }
};

};  // namespace gobonline
