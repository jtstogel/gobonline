#include "src/stepper_error.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "src/util/status_test_utils.h"

namespace gobonline {

TEST(DiscreteUniformIntervalsError, NopWhenModelDoesNothing) {
  DiscreteUniformIntervalsError<4, 0., 8.> error({0, 0, 0});
  EXPECT_DOUBLE_EQ(error.AddError(1.2), 1.2);
  EXPECT_DOUBLE_EQ(error.RemoveError(1.2), 1.2);
}

TEST(DiscreteUniformIntervalsError, ScalesAngleInErrorInterval) {
  // Even:
  // |--|--|--|--|
  //
  // With error:
  // |-|----|--|-|
  DiscreteUniformIntervalsError<4, 0., 8.> error({-1, 1, 1});

  std::vector<double> with_errors = {0., 0.25, 0.5, 0.75, 1., 2.,
                                     3., 4.,   5.,  5.5,  6., 6.5,
                                     7., 7.25, 7.5, 7.75, 8};
  for (int i = 0; i < with_errors.size(); i++) {
    double no_error = 0.5 * static_cast<double>(i);
    double with_error = with_errors[i];
    EXPECT_DOUBLE_EQ(error.AddError(no_error), with_error)
        << "no_error=" << no_error;
    EXPECT_DOUBLE_EQ(error.RemoveError(with_error), no_error)
        << "no_error=" << no_error;
  }
}

}  // namespace gobonline
