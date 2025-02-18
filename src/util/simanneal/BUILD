package(default_visibility = ["//visibility:public"])

cc_library(
    name = "simanneal",
    hdrs = ["simanneal.h"],
    deps = [
        "@abseil-cpp//absl/log:die_if_null",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_test(
    name = "simanneal_test",
    srcs = ["simanneal_test.cc"],
    deps = [
        ":simanneal",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
