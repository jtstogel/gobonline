cc_library(
    name = "goban_cv",
    srcs = ["goban_cv.cc"],
    hdrs = ["goban_cv.h"],
    deps = [
        ":board_state_cc_proto",
        ":cv_macros",
        "//src/util:status",
        "//src/util:union_find",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@opencv-bazel//:opencv",
    ],
)

cc_library(
    name = "cv_macros",
    hdrs = ["cv_macros.h"],
)

cc_test(
    name = "goban_cv_test",
    srcs = [
        "goban_cv_test.cc",
    ],
    data = ["//src/testdata:images"],
    deps = [
        ":goban_cv",
        "//src/testdata:test_image_cc_proto",
        "//src/util:status_test_utils",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@opencv-bazel//:opencv",
        "@protobuf",
    ],
)

proto_library(
    name = "board_state_proto",
    srcs = ["board_state.proto"],
)

cc_proto_library(
    name = "board_state_cc_proto",
    deps = [":board_state_proto"],
)
