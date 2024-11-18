cc_library(
    name = "goban_cv",
    srcs = ["goban_cv.cc"],
    hdrs = ["goban_cv.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@opencv-bazel//:opencv",
    ],
)

cc_test(
    name = "goban_cv_test",
    srcs = [
        "goban_cv_test.cc",
    ],
    data = ["//src/testdata:images"],
    deps = [
        ":goban_cv",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@opencv-bazel//:opencv",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)
