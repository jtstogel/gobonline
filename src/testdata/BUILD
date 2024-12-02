package(default_visibility = ["//visibility:public"])

filegroup(
    name = "images",
    srcs = glob(["*.jpeg", "*.jpg"]) + ["test_images.textproto"],
)

proto_library(
    name = "test_image_proto",
    srcs = ["test_image.proto"],
)

cc_proto_library(
    name = "test_image_cc_proto",
    deps = [":test_image_proto"],
)
