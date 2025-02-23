#include "src/util/simanneal/simanneal.h"

#include <array>
#include <cstddef>
#include <numbers>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace simanneal {

namespace {

using SA = SimulatedAnnealingOptimizer<2, double>;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::Le;
using ::testing::Matcher;

Matcher<double> Between(double lb, double ub) { return AllOf(Ge(lb), Le(ub)); }

}  // namespace

TEST(SimulatedAnnealing, Temperature) {
  double initial_temp = 1000;
  SA::VisitingTemperature t(initial_temp, 2.5);
  EXPECT_NEAR(t(0), initial_temp, 1e-4);
  EXPECT_LT(t(1), t(0));
  EXPECT_NEAR(t(100000), 0., 0.1);
}

TEST(SimulatedAnnealing, AcceptanceProability) {
  SA::VisitingTemperature t(1000., 2.5);
  double acceptance_param = -5;
  SA::AcceptanceProbability p(acceptance_param);

  for (const int iteration : {0, 1, 10}) {
    // Reduction in error is always accepted.
    EXPECT_DOUBLE_EQ(p(/*error_delta=*/-1, t(iteration)), 1.);

    // Positive error delta should have probability [0, 1).
    EXPECT_THAT(p(/*error_delta=*/1, t(iteration)), Between(0, 1 - 1e-4));
  }

  // Positive error delta should have probability [0, 1).
  // Verify that early on we have a non-zero chance
  // of accepting a positive change in error.
  EXPECT_THAT(p(/*error_delta=*/1, /*iteration=*/t(0)),
              Between(1e-4, 1 - 1e-4));
}

TEST(SimulatedAnnealing, VisitingDistributionIsWithinBounds) {
  using SA = SimulatedAnnealingOptimizer<1, double>;
  SA::VisitingTemperature t(1000., 2.5);
  std::array<std::array<double, 2>, 1> bounds({{{-1, 1.}}});
  SA::VisitingDistribution d(2.5, bounds);

  std::array<double, 1> x = {0.};

  absl::BitGen gen;
  std::array<double, 1> x_to_visit = d(gen, x, t(0));
  for (int i = 0; i < 100; i++) {
    EXPECT_THAT(x_to_visit[0], Between(-1., 1.));
  }
}

TEST(SimulatedAnnealing, Simple1DMinimize) {
  using SA = SimulatedAnnealingOptimizer<1>;
  SA::Config config = {
      .bounds = {{{-100., 100.}}},
  };
  SA optimizer(config);

  absl::BitGen gen;
  absl::StatusOr<SA::Result> result = optimizer.Minimize(
      gen,
      [](const std::array<double, 1>& x) { return std::abs(x[0] * x[0] - 2); });

  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_NEAR(result->x[0] * result->x[0], 2, 1e-2);
}

TEST(SimulatedAnnealing, Simple2DWithManyLocalMinima) {
  using SA = SimulatedAnnealingOptimizer<2>;
  SA::Config config = {
      .bounds = {{{-100, 100}, {-100, 100}}},
      // Higher initial temp for a more difficult function.
      .initial_temperature = 10000,
      .max_iterations = 10000,
  };
  SA optimizer(config);

  absl::BitGen gen;
  absl::StatusOr<SA::Result> result =
      optimizer.Minimize(gen, [](const std::array<double, 2>& x) {
        // A bumpy function with lots of local minima.
        return std::sin(3 * x[0]) * std::sin(3 * x[1]) +
               0.1 * (x[0] * x[0] + x[1] * x[1]);
      });

  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_NEAR(result->error, -.946, 1e-2);
}

template <size_t N>
double Ackley(const std::array<double, N>& x, double a = 20.0, double b = 0.2,
              double c = 2.0 * std::numbers::pi) {
  const size_t n = x.size();
  double sum_sq = 0.0;
  for (double xi : x) {
    sum_sq += xi * xi;
  }
  double sum_cos = 0.0;
  for (double xi : x) {
    sum_cos += std::cos(c * xi);
  }
  double term1 = -a * std::exp(-b * std::sqrt(sum_sq / static_cast<double>(n)));
  double term2 = -std::exp(sum_cos / static_cast<double>(n));
  return term1 + term2 + a + std::numbers::e;
}

template <size_t N>
void TestAckley() {
  using SA = SimulatedAnnealingOptimizer<N>;
  std::array<std::array<double, 2>, N> bounds;
  for (size_t i = 0; i < N; i++) {
    bounds[i] = {{-32.768, 32.768}};
  }
  SA optimizer({.bounds = bounds});

  absl::BitGen gen;
  absl::StatusOr<typename SA::Result> result = optimizer.Minimize(
      gen, [](const std::array<double, N>& x) { return Ackley(x); });

  ASSERT_TRUE(result.ok()) << result.status().message();

  std::vector<testing::Matcher<double>> x_matchers;
  x_matchers.reserve(N);
  for (size_t i = 0; i < N; i++) {
    x_matchers.push_back(DoubleNear(0., 1e-2));
  }
  EXPECT_THAT(result->x, testing::ElementsAreArray(x_matchers));
}

TEST(SimulatedAnnealing, Ackley2) { TestAckley<2>(); }

TEST(SimulatedAnnealing, Ackley5) { TestAckley<5>(); }

TEST(SimulatedAnnealing, Ackley10) { TestAckley<10>(); }

TEST(SimulatedAnnealing, Ackley100) { TestAckley<100>(); }

template <size_t N>
double Schwefel(const std::array<double, N>& x) {
  double sum = 0.0;
  for (double xi : x) {
    sum += xi * std::sin(std::sqrt(std::fabs(xi)));
  }
  return 418.9829 * static_cast<double>(x.size()) - sum;
}

template <size_t N>
void TestSchwefel() {
  using SA = SimulatedAnnealingOptimizer<N>;
  std::array<std::array<double, 2>, N> bounds;
  for (size_t i = 0; i < N; i++) {
    bounds[i] = {{-500, 500}};
  }
  SA optimizer({
      .bounds = bounds,
      .max_iterations = 5000,
  });

  absl::BitGen gen;
  absl::StatusOr<typename SA::Result> result =
      optimizer.Minimize(gen, Schwefel<N>);

  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_NEAR(result->error, 0., 1e-2);

  std::vector<testing::Matcher<double>> x_matchers;
  x_matchers.reserve(N);
  for (size_t i = 0; i < N; i++) {
    x_matchers.push_back(testing::DoubleNear(420.9687, 1e-2));
  }
  EXPECT_THAT(result->x, testing::ElementsAreArray(x_matchers));
}

TEST(SimulatedAnnealing, Schwefel2) { TestSchwefel<2>(); }

TEST(SimulatedAnnealing, Schwefel5) { TestSchwefel<5>(); }

TEST(SimulatedAnnealing, Schwefel10) { TestSchwefel<10>(); }

TEST(SimulatedAnnealing, Schwefel100) { TestSchwefel<100>(); }

double Eggholder(const std::array<double, 2> x) {
  double term1 =
      -(x[1] + 47.0) * std::sin(std::sqrt(std::fabs(x[0] / 2.0 + x[1] + 47.0)));
  double term2 = -x[0] * std::sin(std::sqrt(std::fabs(x[0] - (x[1] + 47.0))));
  return term1 + term2;
}

TEST(SimulatedAnnealing, Eggholder) {
  using SA = SimulatedAnnealingOptimizer<2>;
  SA optimizer(SA::Config{
      .bounds = {{{-512, 512}, {-512, 512}}},
      .initial_temperature = 500000,
      .max_iterations = 100000,
  });

  absl::BitGen gen;
  absl::StatusOr<typename SA::Result> result =
      optimizer.Minimize(gen, Eggholder);

  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_THAT(
      result->x,
      AnyOf(
          // The global min.
          ElementsAre(DoubleNear(512, .1), DoubleNear(404.2319, .1)),
          // Other local minima nearby the global min.
          // Too many iterations are required to reach the global min in tests.
          ElementsAre(DoubleNear(482.35, .1), DoubleNear(432.88, .1)),
          ElementsAre(DoubleNear(439.48, .1), DoubleNear(453.98, .1))));
}

}  // namespace simanneal
