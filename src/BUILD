cc_library(
    name = "goban_cv",
    srcs = ["goban_cv.cc"],
    hdrs = ["goban_cv.h"],
    deps = [
        ":board_state_cc_proto",
        ":cv_macros",
        "//src/util:status",
        "//src/util:union_find",
        "//src/util/simanneal",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/random",
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

cc_library(
    name = "laser_calibration_solver",
    srcs = ["laser_calibration_solver.cc"],
    hdrs = ["laser_calibration_solver.h"],
    deps = [
        "//src/util:status",
        "//src/util:union_find",
        "//src/util/simanneal",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen",
    ],
)

cc_test(
    name = "laser_calibration_solver_test",
    srcs = [
        "laser_calibration_solver_test.cc",
    ],
    deps = [
        ":laser_calibration_solver",
        "//src/util:status_test_utils",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:mocking_bit_gen",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@eigen",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
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

cc_library(
    name = "serial_port",
    srcs = ["serial_port.cc"],
    hdrs = ["serial_port.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "//src/util:status",
        "@abseil-cpp//absl/strings",
    ],
)

cc_binary(
    name = "serial_port_main",
    srcs = ["serial_port_main.cc"],
    deps = [
        ":serial_port",
        "//src/util:status",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/flags:parse",
    ],
)

cc_library(
    name = "arduino_controller",
    srcs = ["arduino_controller.cc"],
    hdrs = ["arduino_controller.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "//src/util:status",
        ":serial_port",
        "@abseil-cpp//absl/strings",
        "@re2",
    ],
)

cc_binary(
    name = "arduino_controller_main",
    srcs = ["arduino_controller_main.cc"],
    deps = [
        ":serial_port",
        ":arduino_controller",
        "//src/util:status",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/flags:parse",
    ],
)
