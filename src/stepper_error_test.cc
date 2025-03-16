#include "src/stepper_error.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/util/status_test_utils.h"

namespace gobonline {

TEST(StepperError, NopWhenModelDoesNothing) {
  StepperError error = {
      .error_per_step = {0, 0, 0},
      .period_offset = 0,
  };
  StepperErrorModel model(error);

  EXPECT_EQ(model.AddError(1.2), 1.2);
  EXPECT_EQ(model.RemoveError(1.2), 1.2);
}

TEST(StepperError, ScalesAngleInErrorInterval) {
  StepperError error = {
      .error_per_step = {-.1 * kRadiansPerStep, .1 * kRadiansPerStep, 0},
      .period_offset = 0,
  };
  StepperErrorModel model(error);

  EXPECT_EQ(model.AddError(kRadiansPerStep), .9 * kRadiansPerStep);
  EXPECT_EQ(model.AddError(2 * kRadiansPerStep), 2.1 * kRadiansPerStep);
  EXPECT_EQ(model.AddError(1.5 * kRadiansPerStep), 1.5 * kRadiansPerStep);
}

}  // namespace gobonline
