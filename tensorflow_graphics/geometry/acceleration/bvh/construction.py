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
"""Construction functions for bounding volume hierarchies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.acceleration.bvh import volumes
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape


def _morton_code(points):
  """Compute Morton codes for sets of 3D points"""
  tfb = tf.bitwise

  # Convert points to 20-bit integer representation
  points -= tf.reduce_min(points, axis=-2, keepdims=True)
  points = safe_ops.safe_signed_div(
      points, tf.reduce_max(points, axis=-2, keepdims=True))
  x = tf.cast(points * (2**20 - 1), tf.int64)
  x = tfb.bitwise_and(x, 0xfffff)

  # Repeatedly split and shift bit representations until each original bit is
  # left-padded with two zeros. Example for input 0xfffff:
  # 0000000000 0000000000 0000000000 0000000000 1111111111 1111111111
  # 0000000000 0000000000 1111111111 0000000000 0000000000 1111111111 shift 20
  # 0000000000 1111100000 0000011111 0000000000 1111100000 0000011111 shift 10
  # 0000110000 0011100001 1000000111 0000110000 0011100001 1000000111 shift  6
  # 0000110000 1100100001 1000011001 0000110000 1100100001 1000011001 shift  2
  # 0010010010 0100100100 1001001001 0010010010 0100100100 1001001001 shift  2

  x = tfb.bitwise_or(tfb.bitwise_and(x, 0x3ff),
                     tfb.left_shift(tfb.bitwise_and(x, 0xffc00), 20))
  x = tfb.bitwise_or(tfb.bitwise_and(x, 0x7c000001f),
                     tfb.left_shift(tfb.bitwise_and(x, 0xf8000003e0), 10))
  x = tfb.bitwise_or(tfb.bitwise_and(x, 0xe001c0038007),
                     tfb.left_shift(tfb.bitwise_and(x, 0x30006000c0018), 6))
  x = tfb.bitwise_or(tfb.bitwise_and(x, 0xc0218043008601),
                     tfb.left_shift(tfb.bitwise_and(x, 0xc00180030006), 2))
  x = tfb.bitwise_or(tfb.bitwise_and(x, 0x41208241048209),
                     tfb.left_shift(tfb.bitwise_and(x, 0x82010402080410), 2))

  # Interleave the padded xyz representations
  x = tfb.bitwise_or(
      x[..., 0],
      tfb.bitwise_or(tfb.left_shift(x[..., 1], 1), tfb.left_shift(x[..., 2],
                                                                  2)))

  return x


def construct_bvh(leaf_volumes, volume_type, k=2, name=None):
  """Construct complete k-ary BVH trees for sets of leaf volumes.

  This operation performs bottom-up construction of BVH trees based on a
  space-filling curve.

  Args:
    leaf_volumes: A tensor of shape `[A1, ... , An, M, N]`, which contains sets
      of M bounding volumes of type `volume_type` in its last two dimensions.
    volume_type: A `BoundingVolumeType` or string equivalent describing the
      bounding volume type.
    k: An integer >= 2 defining the branching factor of the tree.
    name: A name for this op. Defaults to "bvh_construct".

  Returns:
    tree: A tensor of shape `[A1, ... , An, T, N]`, which contains the nodes of
      each tree flattened into the second-last dimension level by level,
      starting with the root node.
    leaf_permutations: A tensor of shape `[A1, ... , An, M]`, storing a
      permutation mapping the last level of the tree to the supplied tensor of
      leaf nodes.

  Raises:
    ValueError: if the last dimension of `volumes` does not match the volume
    type, or the volume type is invalid.
  """
  with tf.compat.v1.name_scope(name, "bvh_construct", [leaf_volumes]):
    leaf_volumes = tf.convert_to_tensor(value=leaf_volumes)
    dofs = volumes.volume_dofs(volume_type)

    shape.check_static(leaf_volumes,
                       has_rank_greater_than=1,
                       has_dim_equals=(-1, dofs),
                       tensor_name="leaf_volumes")

    n_leaves = tf.shape(leaf_volumes)[-2]
    batch_shape = tf.shape(leaf_volumes)[:-2]
    n_trees = tf.reduce_prod(batch_shape)

    nodes = tf.reshape(leaf_volumes, (-1, n_leaves, dofs))
    centers = volumes.centers(nodes, volume_type)
    sort_keys = _morton_code(centers)
    leaf_permutations = tf.argsort(sort_keys, axis=-1)
    nodes = tf.gather(nodes, leaf_permutations, axis=-2, batch_dims=1)

    level = tf.cast(
        tf.math.ceil(
            tf.math.log(tf.cast(n_leaves, tf.float32)) / tf.math.log(float(k))),
        tf.int32) - 1

    remainder = k**(level + 1) - n_leaves
    duplicates = tf.tile(nodes[:, -1:], (1, remainder, 1))
    prev_level = tf.concat((nodes, duplicates), axis=-2)

    def _cond(level, nodes, prev_level):
      """Check if the iteration has reached the root node."""
      del nodes, prev_level
      return level >= 0

    def _body(level, nodes, prev_level):
      """Group every k nodes from the previous tree level into a new level."""
      children = tf.reshape(prev_level, (n_trees, k**level, k, dofs))
      new_level = volumes.reduce_union(children, volume_type)
      nodes = tf.concat((new_level, nodes), axis=-2)
      return level - 1, nodes, new_level

    _, nodes, _ = tf.while_loop(_cond,
                                _body, [level, nodes, prev_level],
                                parallel_iterations=1)

    nodes = tf.reshape(nodes,
                       tf.concat((batch_shape, tf.shape(nodes)[1:]), axis=0))
    leaf_permutations = tf.reshape(
        leaf_permutations,
        tf.concat((batch_shape, tf.shape(leaf_permutations)[1:]), axis=0))

    return nodes, leaf_permutations


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
