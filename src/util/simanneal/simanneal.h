/**
 * Implementation of simulated annealing.
 *
 * This is based on SciPy's dual_annealing optimizer.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <ostream>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"

namespace simanneal {

enum class Verbosity {
  kSilent = 0,
  kInfo = 1,
  kDebug = 2,
};

template <typename T, size_t N>
struct RealArray {
  const std::array<T, N>& arr;
};

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const RealArray<T, N>& wrap) {
  os << "[";
  for (size_t i = 0; i < wrap.arr.size(); i++) {
    if (i > 0) os << ", ";
    os << wrap.arr[i];
  }
  os << "]";
  return os;
}

/**
 * Generalized simulated annealing.
 */
template <size_t Dimensions, typename Real = double>
class SimulatedAnnealingOptimizer {
 private:
  using Input = std::array<Real, Dimensions>;
  using Bounds = std::array<std::array<Real, 2>, Dimensions>;
  using ObjectiveFun = absl::AnyInvocable<Real(const Input& x)>;

 public:
  struct Result {
    /* The optimized value of `x`. */
    Input x;

    /* The optimized value of the objective function. */
    Real error;

    /* The number of iterations performed. */
    int iters;
  };

  struct Config {
    /**
     * Bounds for the input.
     *
     * Each bound must present and a finite real.
     */
    Bounds bounds;

    /**
     * The parameter defining the distribution of Δx at every step.
     *
     * Values must be within `(2, 3]`. Larger values will result in
     * more expansive searches at each step.
     */
    Real visiting_param = 2.62;

    /**
     * The parameter defining the probability that we accept
     * uphill values of Δx. Value must be less than 1e-2.
     */
    Real acceptance_param = -5;

    /**
     * Initial temperature of the system.
     */
    Real initial_temperature = 5000.;

    /**
     * Initial value of `x`.
     *
     * If not specified, a random value will be used.
     */
    std::optional<Input> x0 = std::nullopt;

    /**
     * Maximum number of iterations.
     */
    int max_iterations = 1000;

    /**
     * How much to log.
     */
    Verbosity verbosity = Verbosity::kSilent;
  };

  explicit SimulatedAnnealingOptimizer(const Config& config)
      : config_(config) {}

  /** Minimizes the value of `f`. */
  absl::StatusOr<Result> Minimize(absl::BitGenRef gen, ObjectiveFun f) {
    VisitingTemperature visiting_temperature(config_.initial_temperature,
                                             config_.visiting_param);
    VisitingDistribution visiting_distribution(config_.visiting_param,
                                               config_.bounds);

    Input x0 = config_.x0.value_or(UniformRandomX(gen, config_.bounds));
    CandidateVisitor candidate_visitor(config_, x0, f(x0));

    int iteration;
    for (iteration = 0; iteration < config_.max_iterations; iteration++) {
      Real t = visiting_temperature(iteration);

      // Iterate once per dimension so that the default max_iterations
      // scales with the dimension of the input.
      for (size_t dim = 0; dim < Dimensions; dim++) {
        Input candidate_x =
            visiting_distribution(gen, candidate_visitor.CurrentX(), t);
        candidate_visitor.Visit(gen, candidate_x, f(candidate_x), iteration, t);
      }

      Input candidate_x_per_dimension =
          visiting_distribution(gen, candidate_visitor.CurrentX(), t);
      for (size_t dim = 0; dim < Dimensions; dim++) {
        Input candidate_x = candidate_visitor.CurrentX();
        candidate_x[dim] = candidate_x_per_dimension[dim];

        candidate_visitor.Visit(gen, candidate_x, f(candidate_x), iteration, t);
      }
    }

    return Result{
        .x = candidate_visitor.BestX(),
        .error = candidate_visitor.BestError(),
        .iters = iteration,
    };
  }

  /** Provides the next value of Δx to visit at every step. */
  class VisitingDistribution {
   public:
    explicit VisitingDistribution(Real visiting_param, const Bounds& bounds)
        : q_(visiting_param), bounds_(bounds) {}

    Input operator()(absl::BitGenRef gen, const Input& x, Real temperature) {
      Input x_to_visit;
      DeltaX(gen, temperature, /*delta_x=*/x_to_visit);

      for (auto& dx : x_to_visit) {
        dx = TruncateStepSize(gen, dx);
      }

      for (size_t i = 0; i < x.size(); i++) {
        x_to_visit[i] += x[i];
        x_to_visit[i] = ConformToBounds(x_to_visit[i], bounds_[i]);
      }

      return x_to_visit;
    }

   private:
    Real TruncateStepSize(absl::BitGenRef gen, Real dx) {
      static constexpr Real kTailLimit = 1e8;
      if (dx > kTailLimit) {
        return absl::Uniform<Real>(gen, 0, kTailLimit);
      }
      if (dx < -kTailLimit) {
        return absl::Uniform<Real>(gen, -kTailLimit, 0);
      }
      return dx;
    }

    Real ConformToBounds(Real x, const std::array<Real, 2>& bounds) {
      static constexpr Real kMinVisitBound = 1e-10;
      Real lower = bounds[0];
      Real upper = bounds[1];
      Real range = upper - lower;
      x = std::fmod(std::fmod(x - lower, range) + range, range) + lower;
      if (std::fabs(x - lower) < kMinVisitBound) {
        return x + kMinVisitBound;
      }
      return x;
    }

    void DeltaX(absl::BitGenRef gen, Real temperature, Input& delta_x) {
      Real sigma_x = SigmaX(temperature);
      std::generate(delta_x.begin(), delta_x.end(), [&]() {
        Real x = sigma_x * Normal(gen);
        Real y = Normal(gen);
        Real density = Power(std::fabs(y), (q_ - 1.) / (3. - q_));
        return x / density;
      });
    }

    Real SigmaX(Real temperature) {
      // TODO(jtstogel): improve speed by factoring these out to save
      // multiplies.
      Real factor1 = Power(temperature, 1 / (q_ - 1.));
      Real factor2 = Power(q_ - 1, 4 - q_);
      Real factor3 = Power(2., (2 - q_) / (q_ - 1));
      Real factor4 = std::sqrt(std::numbers::pi) * factor1 * factor2 /
                     (factor3 * (3 - q_));
      Real factor5 = 1. / (q_ - 1.) - 0.5;
      Real factor6 = std::numbers::pi * (1. - factor5) /
                     std::sin(std::numbers::pi * (1. - factor5)) /
                     std::exp(std::lgammal(2. - factor5));
      return Power(factor6 / factor4, -(q_ - 1.) / (3. - q_));
    }

    Real Normal(absl::BitGenRef gen) {
      return absl::Gaussian<Real>(gen, /*mean=*/0, /*stddev=*/1);
    }

    const Real q_;
    const Bounds& bounds_;
  };

  /**
   * A temperature source. In general, a temperature is a value
   * that decreases over time to zero.
   *
   * This temperature is computed as:
   *
   *   T(i) = t0 * F(0) / F(1)
   *   F(i) = (i + 1)^p - 1
   */
  class VisitingTemperature {
   public:
    explicit VisitingTemperature(Real t0, Real visiting_param)
        : t0_(t0), visiting_param_(visiting_param), f0_(F(0)) {}

    /** Returns the temperature at `iteration` (zero-indexed). */
    Real operator()(int iteration) const { return t0_ * f0_ / F(iteration); }

    Real F(int iteration) const {
      return Power(2. + static_cast<Real>(iteration), visiting_param_ - 1.) -
             1.;
    }

    const Real t0_;
    const Real visiting_param_;
    // F(0), Stored to avoid recomputation.
    const Real f0_;
  };

  /**
   * The probability of accepting a new value E(x).
   */
  class AcceptanceProbability {
   public:
    explicit AcceptanceProbability(Real acceptance_param)
        : acceptance_param_(acceptance_param),
          exponent_(1 / (1 - acceptance_param_)) {}

    Real operator()(Real error_delta, Real acceptance_temperature) const {
      if (error_delta < 0) {
        return 1.;  // Always accept values that decrease error.
      }

      double base =
          1. - (1. - acceptance_param_) * error_delta / acceptance_temperature;
      if (base <= 0) {
        return 0.;
      }

      return Power(base, exponent_);
    }

   private:
    const Real acceptance_param_;
    const Real exponent_;  // Precomputed value.
  };

  class CandidateVisitor {
   public:
    explicit CandidateVisitor(const Config& config, const Input& x, Real error)
        : config_(config),
          acceptance_probability_(config.acceptance_param),
          current_x_(x),
          current_error_(error),
          best_x_(x),
          best_error_(error) {}

    void Visit(absl::BitGenRef gen, const Input& x, Real error, int iteration,
               Real visiting_temperature) {
      if (config_.verbosity >= Verbosity::kDebug) {
        std::cerr << "Iteration: " << iteration << "\n"                   //
                  << "  Visiting temp: " << visiting_temperature << "\n"  //
                  << "  Current X: " << RealArray{current_x_} << "\n"     //
                  << "  Current error: " << current_error_ << "" << "\n"  //
                  << "  Candidate X: " << RealArray{x} << "\n"            //
                  << "  Candidate error: " << error << "" << "\n"         //
                  << "  Best X: " << RealArray{best_x_} << "\n"           //
                  << "  Best error: " << best_error_ << "" << "\n"        //
                  << std::endl;
      }

      Real acceptance_temperature = visiting_temperature / (iteration + 1);
      if (ShouldAccept(gen, error, acceptance_temperature)) {
        current_x_ = x;
        current_error_ = error;
      }

      if (current_error_ < best_error_) {
        best_x_ = current_x_;
        best_error_ = current_error_;
      }
    }

    const Input& CurrentX() { return current_x_; }

    const Input& BestX() { return best_x_; }

    Real BestError() { return best_error_; }

   private:
    bool ShouldAccept(absl::BitGenRef gen, Real error,
                      Real acceptance_temperature) const {
      double p = acceptance_probability_(error - current_error_,
                                         acceptance_temperature);
      if (p == 0.) return false;
      if (p == 1.) return true;
      return absl::Uniform<Real>(gen, 0., 1.) < p;
    }

    const Config& config_;
    AcceptanceProbability acceptance_probability_;

    Input current_x_;
    Real current_error_;

    Input best_x_;
    Real best_error_;
  };

  static Real Power(Real value, Real exponent) {
    return std::exp(exponent * std::log(value));
  }

  Input UniformRandomX(absl::BitGenRef gen, const Bounds& bounds) {
    Input x;
    for (size_t i = 0; i < x.size(); i++) {
      x[i] = absl::Uniform<Real>(gen, bounds[i][0], bounds[i][1]);
    }
    return x;
  }

 private:
  const Config config_;
};

}  // namespace simanneal
