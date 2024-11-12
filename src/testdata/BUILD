package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_image_files",
    srcs = glob(["*.jpeg"]) + ["test_images.textproto"],
)

proto_library(
    name = "test_image_proto",
    srcs = ["test_image.proto"],
)

cc_proto_library(
    name = "test_image_cc_proto",
    deps = [":test_image_proto"],
)
