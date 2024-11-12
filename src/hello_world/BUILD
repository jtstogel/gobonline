cc_binary(
    name = "hello_world",
    srcs = ["hello_world.cc"],
)

cc_test(
    name = "hello_world_test",
    size = "small",
    srcs = ["hello_world_test.cc"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
