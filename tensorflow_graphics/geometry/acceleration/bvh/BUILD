#Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Bounding volume hierarchy operations for tf-graphics.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "bvh",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":construction",
        ":volumes",
        "//tensorflow_graphics/util:doc",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "volumes",
    srcs = ["volumes.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "construction",
    srcs = ["construction.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":volumes",
        "//tensorflow_graphics/util:export_api",
    ],
)
