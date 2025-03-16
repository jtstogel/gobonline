#include "src/stepper_error.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

namespace gobonline {
namespace {

double PositiveMod(double n, double d) {
  return std::fmod(std::fmod(n, d) + d, d);
}

}  // namespace

StepperErrorModel::StepperErrorModel(StepperError error)
    : error_(std::move(error)) {
  start_angles_[0] = 0;
  for (int i = 1; i < kStepsPerElectricalErrorPeriod; i++) {
    start_angles_[i] = i * kRadiansPerStep + error_.error_per_step[i - 1];
  }

  for (int i = 0; i < kStepsPerElectricalErrorPeriod - 1; i++) {
    intervals_[i] = start_angles_[i + 1] - start_angles_[i];
  }

  intervals_[kStepsPerElectricalErrorPeriod-1] =
      kRadiansPerErrorPeriod - intervals_[kStepsPerElectricalErrorPeriod - 1];
}

double StepperErrorModel::AddError(double angle) const {
  double offset =
      PositiveMod(angle + (error_.period_offset * kRadiansPerErrorPeriod),
                  kRadiansPerErrorPeriod);

  double step_index = static_cast<int>(offset / kRadiansPerStep);
  double t = std::fmod(offset, kRadiansPerStep) / kRadiansPerStep;

  double offset_within_error_period =
      start_angles_[step_index] + (t * intervals_[step_index]);

  return angle - offset + offset_within_error_period;
}

double StepperErrorModel::RemoveError(double angle) const {
  (void)angle;
  (void)start_angles_;
  return 0.;
}

std::pair<int, double> StepperErrorModel::GetStepIndexAndPosition(
    double angle_with_offset) const {
  int step_index = 0;
  for (int i = 0; i < kStepsPerElectricalErrorPeriod; ++i) {
    if (angle_with_offset >= start_angles_[i]) {
      step_index = i;
      break;
    }
  }
  double t =
      (angle_with_offset - start_angles_[step_index]) / intervals_[step_index];
  return {step_index, t};
}

}  // namespace gobonline
