/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow_graphics/geometry/representation/mesh/marching_cubes.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct CellTriangleCount<CPUDevice, T> {
  void operator()(const CPUDevice& device, const GridType<T>& grid,
                  CountsType* cell_counts, float isolevel) const {
    auto op = [&](int64 start, int64 end) {
      for (int64 i = start; i < end; ++i) {
        (*cell_counts)(i) = CountTrianglesInCell<T>(i, grid, isolevel);
      }
    };
    // TODO(drebain) compute cost correctly
    device.parallelFor(cell_counts->size(), Eigen::TensorOpCost(1.0, 1.0, 0),
                       op);
  }
};

template <>
struct CumulativeSum<CPUDevice> {
  int64 operator()(const CPUDevice& device, CountsType* cell_counts) const {
    int64 prev;
    for (int64 i = 0; i < cell_counts->size(); ++i) {
      int64 temp = prev;
      prev = (*cell_counts)(i);
      if (i == 0) {
        (*cell_counts)(i) = 0;
      } else {
        (*cell_counts)(i) = (*cell_counts)(i - 1) + temp;
      }
    }
    return prev + (*cell_counts)(cell_counts->size() - 1);
  }
};

template <>
struct TriangleIndexScatter<CPUDevice> {
  void operator()(const CPUDevice& device, const CountsType& cell_counts,
                  TrianglesIndicesType* triangle_indices) const {
    int64 total_triangles = triangle_indices->dimension(0);
    auto op = [&](int64 start, int64 end) {
      for (int64 i = start; i < end; ++i) {
        int64 cell_i_triangles;
        if (i < cell_counts.size() - 1) {
          cell_i_triangles = cell_counts(i + 1) - cell_counts(i);
        } else {
          cell_i_triangles = total_triangles - cell_counts(i);
        }
        for (int64 j = 0; j < cell_i_triangles; ++j) {
          // Which cell is this triangle in?
          (*triangle_indices)(cell_counts(i) + j, 0) = i;

          // Which of the triangles in the cell is this?
          (*triangle_indices)(cell_counts(i) + j, 1) = j;
        }
      }
    };
    // TODO(drebain) compute cost correctly
    device.parallelFor(cell_counts.size(), Eigen::TensorOpCost(1.0, 1.0, 0),
                       op);
  }
};

template <typename T>
struct TriangleBuilder<CPUDevice, T> {
  void operator()(const CPUDevice& device, const GridType<T>& grid,
                  const TrianglesIndicesType& triangle_indices,
                  TrianglesType<T>* triangles, float isolevel) const {
    auto op = [&](int64 start, int64 end) {
      for (int64 i = start; i < end; ++i) {
        auto tri = ComputeTriangle<T>(i, grid, triangle_indices, isolevel);
        for (int64 j = 0; j < 3; ++j) {
          for (int64 k = 0; k < 3; ++k) {
            (*triangles)(i, j, k) = tri(j, k);
          }
        }
      }
    };
    // TODO(drebain) compute cost correctly
    device.parallelFor(triangle_indices.dimension(0),
                       Eigen::TensorOpCost(1.0, 1.0, 0), op);
  }
};

}  // namespace functor

template <typename Device, typename T>
class MarchingCubesOp : public tensorflow::OpKernel {
 public:
  explicit MarchingCubesOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("isolevel", &isolevel));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& grid_tensor = context->input(0);

    // Count triangles per cell.
    Tensor cell_counts_tensor;
    TensorShape grid_shape = grid_tensor.shape();
    TensorShape cells_shape({grid_shape.dim_size(0) - 1,
                             grid_shape.dim_size(1) - 1,
                             grid_shape.dim_size(2) - 1});
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataType::DT_INT64, cells_shape,
                                          &cell_counts_tensor));
    auto cell_counts = cell_counts_tensor.tensor<int64, 3>();
    const auto& grid = grid_tensor.tensor<T, 3>();
    functor::CellTriangleCount<Device, T>()(context->eigen_device<Device>(),
                                            grid, &cell_counts, isolevel);

    // Reduce output indices for each cell.
    int64 total_triangles = functor::CumulativeSum<Device>()(
        context->eigen_device<Device>(), &cell_counts);

    // Build array of triangle indices: (cell_index, subtriangle_index).
    Tensor triangle_indices_tensor;
    TensorShape triangle_indices_shape({total_triangles, 2});
    OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT64,
                                                   triangle_indices_shape,
                                                   &triangle_indices_tensor));
    auto triangle_indices = triangle_indices_tensor.tensor<int64, 2>();
    functor::TriangleIndexScatter<Device>()(context->eigen_device<Device>(),
                                            cell_counts, &triangle_indices);

    // Compute triangles from grid values.
    Tensor* triangles_tensor;
    TensorShape triangles_shape({total_triangles, 3, 3});
    OP_REQUIRES_OK(context, context->allocate_output(0, triangles_shape,
                                                     &triangles_tensor));
    auto triangles = triangles_tensor->tensor<T, 3>();
    functor::TriangleBuilder<Device, T>()(context->eigen_device<Device>(), grid,
                                          triangle_indices, &triangles,
                                          isolevel);
  }

 private:
  float isolevel;
};

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubes")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<double>("T"),
    tensorflow::MarchingCubesOp<tensorflow::CPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubes")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<float>("T"),
    tensorflow::MarchingCubesOp<tensorflow::CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubes")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<Eigen::half>("T"),
    tensorflow::MarchingCubesOp<tensorflow::CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubes")
        .Device(tensorflow::DEVICE_GPU)
        .TypeConstraint<double>("T"),
    tensorflow::MarchingCubesOp<tensorflow::GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubes")
        .Device(tensorflow::DEVICE_GPU)
        .TypeConstraint<float>("T"),
    tensorflow::MarchingCubesOp<tensorflow::GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubes")
        .Device(tensorflow::DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T"),
    tensorflow::MarchingCubesOp<tensorflow::GPUDevice, Eigen::half>);

REGISTER_OP("MarchingCubes")
    .Input("field_values: T")
    .Attr("isolevel: float")
    .Attr("T: {float16, float32, float64}")
    .Output("triangles: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using shape_inference::ShapeHandle;
      ShapeHandle grid;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grid));
      c->set_output(
          0, c->MakeShape({c->UnknownDim(), c->MakeDim(3), c->MakeDim(3)}));
      return Status::OK();
    });

}  // namespace tensorflow
