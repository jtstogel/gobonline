#pragma once

#include <Eigen/Core>
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

class StepperErrorModel {
 public:
  explicit StepperErrorModel(StepperError error);
  double AddError(double angle) const;
  double RemoveError(double angle) const;

 private:
  using PeriodVec = Eigen::Vector<double, kStepsPerElectricalErrorPeriod>;

  StepperError error_;
  PeriodVec intervals_ = PeriodVec::Zero();
  PeriodVec start_angles_ = PeriodVec::Zero();

  PeriodVec ComputeStartAngles() const;

  std::pair<int, double> GetStepIndexAndPosition(
      double angle_with_offset) const;
};

};  // namespace gobonline
