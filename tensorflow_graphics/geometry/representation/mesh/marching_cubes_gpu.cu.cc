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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_graphics/geometry/representation/mesh/marching_cubes.h"
#include "third_party/cub/device/device_scan.cuh"

namespace gpuprim = ::cub;

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
__global__ void CellTriangleCountKernel(const GridType<T> grid,
                                        CountsType cell_counts,
                                        float isolevel) {
  for (int i : GpuGridRangeX(cell_counts.size())) {
    cell_counts(i) = CountTrianglesInCell<T>(i, grid, isolevel);
  }
}

template <typename T>
void CellTriangleCount<GPUDevice, T>::operator()(const GPUDevice& device,
                                                 const GridType<T>& grid,
                                                 CountsType* cell_counts,
                                                 float isolevel) const {
  GpuLaunchConfig config = GetGpuLaunchConfig(cell_counts->size(), device);
  TF_CHECK_OK(GpuLaunchKernel(CellTriangleCountKernel<T>, config.block_count,
                              config.thread_per_block, 0, device.stream(), grid,
                              *cell_counts, isolevel));
}

int64 CumulativeSum<GPUDevice>::operator()(const GPUDevice& device,
                                           CountsType* cell_counts) const {
  // Retrieve the last count entry before overwriting it.
  int64 last_entry;
  int64* counts_begin = cell_counts->data();
  device.memcpyDeviceToHost(
      &last_entry, &(counts_begin[cell_counts->size() - 1]), sizeof(int64));

  // Get required temporary storage for CUB operation.
  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  auto result = gpuprim::DeviceScan::ExclusiveSum(
      temp_storage, temp_storage_bytes, counts_begin, counts_begin,
      cell_counts->size(), device.stream());
  if (result != gpuSuccess) {
    TF_CHECK_OK(errors::Internal(GpuGetErrorString(result)));
  }
  temp_storage = device.allocate_temp(temp_storage_bytes);

  // Perform parallel prefix sum.
  result = gpuprim::DeviceScan::ExclusiveSum(
      temp_storage, temp_storage_bytes, counts_begin, counts_begin,
      cell_counts->size(), device.stream());
  if (result != gpuSuccess) {
    TF_CHECK_OK(errors::Internal(GpuGetErrorString(result)));
  }
  device.deallocate_temp(temp_storage);

  // Retrieve the last count entry, now containing the sum.
  int64 sum;
  device.memcpyDeviceToHost(&sum, &(counts_begin[cell_counts->size() - 1]),
                            sizeof(int64));

  return sum + last_entry;
}

__global__ void TriangleIndexScatterKernel(
    const CountsType cell_counts, TrianglesIndicesType triangle_indices) {
  int64 total_triangles = triangle_indices.dimension(0);
  for (int i : GpuGridRangeX(cell_counts.size())) {
    int64 cell_i_triangles;
    if (i < cell_counts.size() - 1) {
      cell_i_triangles = cell_counts(i + 1) - cell_counts(i);
    } else {
      cell_i_triangles = total_triangles - cell_counts(i);
    }
    for (int64 j = 0; j < cell_i_triangles; ++j) {
      // Which cell is this triangle in:
      triangle_indices(cell_counts(i) + j, 0) = i;
      // Which of the triangles in the cell is this:
      triangle_indices(cell_counts(i) + j, 1) = j;
    }
  }
}

void TriangleIndexScatter<GPUDevice>::operator()(
    const GPUDevice& device, const CountsType& cell_counts,
    TrianglesIndicesType* triangle_indices) const {
  GpuLaunchConfig config = GetGpuLaunchConfig(cell_counts.size(), device);
  TF_CHECK_OK(GpuLaunchKernel(TriangleIndexScatterKernel, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              cell_counts, *triangle_indices));
}

template <typename T>
__global__ void TriangleBuilderKernel(
    const GridType<T> grid, const TrianglesIndicesType triangle_indices,
    TrianglesType<T> triangles, float isolevel) {
  for (int i : GpuGridRangeX(triangle_indices.dimension(0))) {
    auto tri = ComputeTriangle<T>(i, grid, triangle_indices, isolevel);
    for (int64 j = 0; j < 3; ++j) {
      for (int64 k = 0; k < 3; ++k) {
        triangles(i, j, k) = tri(j, k);
      }
    }
  }
}

template <typename T>
void TriangleBuilder<GPUDevice, T>::operator()(
    const GPUDevice& device, const GridType<T>& grid,
    const TrianglesIndicesType& triangle_indices, TrianglesType<T>* triangles,
    float isolevel) const {
  GpuLaunchConfig config = GetGpuLaunchConfig(
      triangle_indices.dimension(0), device, TriangleBuilderKernel<T>, 0, 1024);
  TF_CHECK_OK(GpuLaunchKernel(TriangleBuilderKernel<T>, config.block_count,
                              config.thread_per_block, 0, device.stream(), grid,
                              triangle_indices, *triangles, isolevel));
}

template struct CellTriangleCount<GPUDevice, double>;
template struct CellTriangleCount<GPUDevice, float>;
template struct CellTriangleCount<GPUDevice, Eigen::half>;

template struct TriangleBuilder<GPUDevice, double>;
template struct TriangleBuilder<GPUDevice, float>;
template struct TriangleBuilder<GPUDevice, Eigen::half>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA