package(default_visibility = ["//visibility:public"])

cc_library(
    name = "status",
    hdrs = ["status.h"],
    deps = [":macro_utils"],
)

cc_library(
    name = "status_test_utils",
    hdrs = ["status_test_utils.h"],
    deps = [":macro_utils"],
)

cc_library(
    name = "macro_utils",
    hdrs = ["macro_utils.h"],
)

cc_library(
    name = "union_find",
    srcs = ["union_find.cc"],
    hdrs = ["union_find.h"],
)

cc_test(
    name = "union_find_test",
    srcs = ["union_find_test.cc"],
    deps = [
        ":union_find",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
