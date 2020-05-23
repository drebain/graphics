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
  void operator()(const CPUDevice& device, const GridTType<T>& grid,
                  CountsTType* cell_counts,
                  const IsolevelTType<T>& isolevel) const {
    auto op = [&](int64 start, int64 end) {
      for (int64 i = start; i < end; ++i) {
        (*cell_counts)(i) = CountTrianglesInCell<T>(i, grid, isolevel());
      }
    };
    // TODO(drebain) compute cost correctly
    device.parallelFor(cell_counts->size(), Eigen::TensorOpCost(1.0, 1.0, 0),
                       op);
  }
};

template <>
struct CumulativeSum<CPUDevice> {
  int64 operator()(const CPUDevice& device, CountsTType* cell_counts) const {
    int64 prev = 0;
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
  void operator()(const CPUDevice& device, const CountsTType& cell_counts,
                  TriangleIndicesTType* triangle_indices) const {
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
struct ComputeTriangles<CPUDevice, T> {
  void operator()(const CPUDevice& device, const GridTType<T>& grid,
                  const TriangleIndicesTType& triangle_indices,
                  TrianglesTType<T>* triangles,
                  const IsolevelTType<T>& isolevel) const {
    auto op = [&](int64 start, int64 end) {
      for (int64 i = start; i < end; ++i) {
        auto tri = ComputeTriangle<T>(i, grid, triangle_indices, isolevel());
        Eigen::Map<TriangleType<T>>(&(*triangles)(i, 0, 0)) = tri;
      }
    };
    // TODO(drebain) compute cost correctly
    device.parallelFor(triangle_indices.dimension(0),
                       Eigen::TensorOpCost(1.0, 1.0, 0), op);
  }
};

template <typename T>
struct ComputeGradients<CPUDevice, T> {
  void operator()(const CPUDevice& device, const GridTType<T>& grid,
                  const TriangleIndicesTType& triangle_indices,
                  const TrianglesGradientTType<T>& triangle_gradients,
                  GridGradientTType<T>* grid_gradients,
                  IsolevelGradientTType<T>* isolevel_gradient,
                  const IsolevelTType<T>& isolevel) {
    device.memset(grid_gradients->data(), 0,
                  sizeof(T) * grid_gradients->size());
    (*isolevel_gradient)() = T(0.0);
    for (int64 i = 0; i < triangle_indices.dimension(0); ++i) {
      auto triangle_gradient =
          Eigen::Map<const TriangleType<T>>(&triangle_gradients(i, 0, 0));
      (*isolevel_gradient)() += ComputeTriangleGradients<T>(
          i, grid, triangle_indices, triangle_gradient, grid_gradients,
          isolevel());
    }
  }
};

}  // namespace functor

template <typename Device, typename T>
class MarchingCubesOp : public tensorflow::OpKernel {
 public:
  explicit MarchingCubesOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& grid_tensor = context->input(0);
    const Tensor& isolevel_tensor = context->input(1);

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
    const auto& isolevel = isolevel_tensor.scalar<T>();
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
    functor::ComputeTriangles<Device, T>()(context->eigen_device<Device>(),
                                           grid, triangle_indices, &triangles,
                                           isolevel);
  }
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
    .Input("isolevel: T")
    .Attr("T: {float16, float32, float64}")
    .Output("triangles: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using shape_inference::ShapeHandle;
      ShapeHandle grid, isolevel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grid));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &isolevel));
      c->set_output(
          0, c->MakeShape({c->UnknownDim(), c->MakeDim(3), c->MakeDim(3)}));
      return Status::OK();
    });

template <typename Device, typename T>
class MarchingCubesGradientOp : public tensorflow::OpKernel {
 public:
  explicit MarchingCubesGradientOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& grid_tensor = context->input(0);
    const Tensor& isolevel_tensor = context->input(1);
    const Tensor& triangle_gradients_tensor = context->input(2);

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
    const auto& isolevel = isolevel_tensor.scalar<T>();
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

    // Compute gradients for grid values and isolevel.
    Tensor* grid_gradients_tensor;
    Tensor* isolevel_gradient_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, grid_tensor.shape(),
                                                     &grid_gradients_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, isolevel_tensor.shape(),
                                            &isolevel_gradient_tensor));
    auto triangle_gradients = triangle_gradients_tensor.tensor<T, 3>();
    auto grid_gradients = grid_gradients_tensor->tensor<T, 3>();
    auto isolevel_gradient = isolevel_gradient_tensor->scalar<T>();
    functor::ComputeGradients<Device, T>()(
        context->eigen_device<Device>(), grid, triangle_indices,
        triangle_gradients, &grid_gradients, &isolevel_gradient, isolevel);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubesGradient")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<double>("T"),
    tensorflow::MarchingCubesGradientOp<tensorflow::CPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubesGradient")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<float>("T"),
    tensorflow::MarchingCubesGradientOp<tensorflow::CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubesGradient")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<Eigen::half>("T"),
    tensorflow::MarchingCubesGradientOp<tensorflow::CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubesGradient")
        .Device(tensorflow::DEVICE_GPU)
        .TypeConstraint<double>("T"),
    tensorflow::MarchingCubesGradientOp<tensorflow::GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubesGradient")
        .Device(tensorflow::DEVICE_GPU)
        .TypeConstraint<float>("T"),
    tensorflow::MarchingCubesGradientOp<tensorflow::GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("MarchingCubesGradient")
        .Device(tensorflow::DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T"),
    tensorflow::MarchingCubesGradientOp<tensorflow::GPUDevice, Eigen::half>);

REGISTER_OP("MarchingCubesGradient")
    .Input("field_values: T")
    .Input("isolevel: T")
    .Input("triangle_gradients: T")
    .Attr("T: {float16, float32, float64}")
    .Output("grid_gradients: T")
    .Output("isolevel_gradient: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using shape_inference::ShapeHandle;
      using shape_inference::DimensionHandle;
      ShapeHandle grid, isolevel, triangle_gradients;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grid));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &isolevel));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &triangle_gradients));
      DimensionHandle verts_per_tri, components_per_vert;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(triangle_gradients, 1), 3, &verts_per_tri));
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(triangle_gradients, 2), 3, &components_per_vert));
      c->set_output(0, grid);
      c->set_output(1, isolevel);
      return Status::OK();
    });

}  // namespace tensorflow
