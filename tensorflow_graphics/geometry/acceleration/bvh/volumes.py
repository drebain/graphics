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
"""Utility functions for manipulating bounding volumes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


class BoundingVolumeType(object):
  """Defines supported bounding volume primitives.

  A single volume of a particular type is encoded as an N-D vector.

  Supported volume types:

  *   <b>`aabb`</b>: (N = 6) An axis-aligned bounding box encoded as
    `[min_x, min_y, min_z, max_x, max_y, max_z]`.
  *   <b>`sphere`</b>: (N = 4) A bounding sphere encoded as
    `[center_x, center_y, center_z, radius]`.
  """
  AABB = "aabb"
  SPHERE = "sphere"


def volume_dofs(volume_type):
  """Returns the number of dimensions needed to represent `volume_type`."""
  if volume_type == BoundingVolumeType.AABB:
    return 6
  elif volume_type == BoundingVolumeType.SPHERE:
    return 4
  else:
    raise ValueError("Unknown bounding volume type supplied.")


def union(volumes1, volumes2, volume_type, name=None):
  """Computes the smallest bounding volumes containing two children.

  Note:
    This does not compute a strict union as defined by set theory. Rather, it
    finds the smallest instance of `volume_type` which is a superset of the
    set-theoretical union.

  Args:
    volumes1: A tensor of shape `[A1, ... , An, N]`, which contains bounding
      volumes of type `volume_type` in its last dimension.
    volumes2: A tensor of shape `[A1, ... , An, N]`, which contains bounding
      volumes of type `volume_type` in its last dimension.
    volume_type: A `BoundingVolumeType` or string equivalent describing the
      bounding volume type.
    name: A name for this op. Defaults to "bv_union".

  Returns:
    A tensor of shape `[A1, ... , An, N]` containing the union volumes.

  Raises:
    ValueError: if the last dimensions of `volumes1` and `volumes2` do not match
    the volume type, or the volume type is invalid.
  """
  with tf.compat.v1.name_scope(name, "bv_union", [volumes1, volumes2]):
    volumes1 = tf.convert_to_tensor(value=volumes1)
    volumes2 = tf.convert_to_tensor(value=volumes2)
    shape.compare_batch_dimensions(tensors=(volumes1, volumes2),
                                   last_axes=(-2, -2),
                                   broadcast_compatible=True,
                                   tensor_names=("volumes1", "volumes2"))
    shape.check_static(volumes1,
                       has_dim_equals=(-1, volume_dofs(volume_type)),
                       tensor_name="volumes1")
    shape.check_static(volumes2,
                       has_dim_equals=(-1, volume_dofs(volume_type)),
                       tensor_name="volumes2")

    if volume_type == BoundingVolumeType.AABB:
      min1, max1 = tf.split(volumes1, 2, axis=-1)
      min2, max2 = tf.split(volumes2, 2, axis=-1)
      union_min = tf.where(min1 < min2, min1, min2)
      union_max = tf.where(max1 > max2, max1, max2)
      union_volumes = tf.concat((union_min, union_max), axis=-1)
    elif volume_type == BoundingVolumeType.SPHERE:
      c1, r1 = tf.split(volumes1, [3, 1], axis=-1)
      c2, r2 = tf.split(volumes2, [3, 1], axis=-1)
      axes, seperations = tf.linalg.normalize(c2 - c1, axis=-1)
      union_c = (c1 + c2 + axes * (r2 - r1)) / 2.0
      union_r = seperations + r1 + r2
      union_volumes = tf.concat((union_c, union_r), axis=-1)
    else:
      raise ValueError("Unknown bounding volume type supplied.")

    return union_volumes


def reduce_union(volumes, volume_type, name=None):
  """Approximates the smallest bounding volumes containing some children.

  Given sets of M bounding volumes, this function computes single volumes which
  contain all M children. Depending on the `volume_type` and inputs given, this
  parent volume may be exactly the smallest possible superset, or may be
  arbitrarily larger.

  Note:
    This does not compute a strict union as defined by set theory. Rather, it
    tries to find the smallest instance of `volume_type` which is a superset
    of the set-theoretical union.

  Args:
    volumes: A tensor of shape `[A1, ... , An, M, N]`, which contains sets of
      M bounding volumes of type `volume_type` in its last two dimensions.
    volume_type: A `BoundingVolumeType` or string equivalent describing the
      bounding volume type.
    name: A name for this op. Defaults to "bv_union".

  Returns:
    A tensor of shape `[A1, ... , An, N]` containing the union volumes.

  Raises:
    ValueError: if the last dimension of `volumes` does not match the volume
    type, or the volume type is invalid.
  """
  with tf.compat.v1.name_scope(name, "bv_union", [volumes]):
    volumes = tf.convert_to_tensor(value=volumes)
    shape.check_static(volumes,
                       has_rank_greater_than=1,
                       has_dim_equals=(-1, volume_dofs(volume_type)),
                       tensor_name="volumes")

    if volume_type == BoundingVolumeType.AABB:
      vmin, vmax = tf.split(volumes, 2, axis=-1)
      union_min = tf.reduce_min(vmin, axis=-2)
      union_max = tf.reduce_max(vmax, axis=-2)
      union_volumes = tf.concat((union_min, union_max), axis=-1)
    elif volume_type == BoundingVolumeType.SPHERE:
      c, r = tf.split(volumes, [3, 1], axis=-1)
      union_c = (tf.reduce_min(c, axis=-2) + tf.reduce_max(c, axis=-2)) / 2.0
      union_r = tf.reduce_max(
          tf.norm(c - union_c[..., None, :], axis=-1, keepdims=True) + r,
          axis=-2)
      union_volumes = tf.concat((union_c, union_r), axis=-1)
    else:
      raise ValueError("Unknown bounding volume type supplied.")

    return union_volumes


def test_intersection(volumes1, volumes2, volume_type, name=None):
  """Test bounding volume pairs for non-empty intersection.

  Args:
    volumes1: A tensor of shape `[A1, ... , An, N]`, which contains bounding
      volumes of type `volume_type` in its last dimension.
    volumes2: A tensor of shape `[A1, ... , An, N]`, which contains bounding
      volumes of type `volume_type` in its last dimension.
    volume_type: A `BoundingVolumeType` or string equivalent describing the
      bounding volume type.
    name: A name for this op. Defaults to "bv_intersection".

  Returns:
    A bool tensor of shape `[A1, ... , An, 1]` indicating which pairs intersect.

  Raises:
    ValueError: if the last dimensions of `volumes1` and `volumes2` do not match
    the volume type, or the volume type is invalid.
  """
  with tf.compat.v1.name_scope(name, "bv_intersection", [volumes1, volumes2]):
    volumes1 = tf.convert_to_tensor(value=volumes1)
    volumes2 = tf.convert_to_tensor(value=volumes2)
    shape.compare_batch_dimensions(tensors=(volumes1, volumes2),
                                   last_axes=(-2, -2),
                                   broadcast_compatible=True,
                                   tensor_names=("volumes1", "volumes2"))
    shape.check_static(volumes1,
                       has_dim_equals=(-1, volume_dofs(volume_type)),
                       tensor_name="volumes1")
    shape.check_static(volumes2,
                       has_dim_equals=(-1, volume_dofs(volume_type)),
                       tensor_name="volumes2")

    if volume_type == BoundingVolumeType.AABB:
      min1, max1 = tf.split(volumes1, 2, axis=-1)
      min2, max2 = tf.split(volumes2, 2, axis=-1)
      # TODO(drebain) implement aabb-aabb intersection
      raise NotImplementedError()
    elif volume_type == BoundingVolumeType.SPHERE:
      c1, r1 = tf.split(volumes1, [3, 1], axis=-1)
      c2, r2 = tf.split(volumes2, [3, 1], axis=-1)
      seperations = tf.linalg.norm(c2 - c1, keepdims=True, axis=-1)
      intersections = seperations < (r1 + r2)
    else:
      raise ValueError("Unknown bounding volume type supplied.")

    return intersections


def centers(volumes, volume_type, name=None):
  """Finds centers of volumes suitable for approximating their positions.

  Args:
    volumes: A tensor of shape `[A1, ... , An, N]`, which contains bounding
      volumes of type `volume_type` in its last dimension.
    volume_type: A `BoundingVolumeType` or string equivalent describing the
      bounding volume type.
    name: A name for this op. Defaults to "bv_center".

  Returns:
    A tensor of shape `[A1, ... , An, 3]` containing the volume centers.

  Raises:
    ValueError: if the last dimension of `volumes` does not match the volume
    type, or the volume type is invalid.
  """
  with tf.compat.v1.name_scope(name, "bv_center", [volumes]):
    volumes = tf.convert_to_tensor(value=volumes)
    shape.check_static(volumes,
                       has_dim_equals=(-1, volume_dofs(volume_type)),
                       tensor_name="volumes")

    if volume_type == BoundingVolumeType.AABB:
      volume_centers = (volumes[..., :3] + volumes[..., 3:]) / 2.0
    elif volume_type == BoundingVolumeType.SPHERE:
      volume_centers = volumes[..., :3]
    else:
      raise ValueError("Unknown bounding volume type supplied.")

    return volume_centers


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
