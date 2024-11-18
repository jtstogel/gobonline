namespace gobonline {

template <typename T>
struct ValueBounds {
  T lower;
  T upper;
};

// Goban intersection lines can be between 0.6 and 1.2 millimeters.
constexpr ValueBounds<double> kGobanLineMillimeterBounds = {
    .lower = 0.6,
    .upper = 1.2,
};

}  // namespace gobonline
