from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.distributions import distribution as distribution_lib
import numpy as np

_MAXINT32 = 2**31 - 1

def truncate_seed(seed):
  return seed % _MAXINT32  # Truncate to fit into 32-bit integer


class IndependentList(distribution_lib.Distribution):
  """Independent distribution from a list of distributions.
  """

  def __init__(
      self, distributions,
      validate_args=False, name=None):
    """  Construct a `IndependentList` distribution.
    """
    parameters = locals()
    name = name or "IndependentList"
    self._distributions = distributions
    self._event_shapes = []
    for d in self._distributions:
        if len(d.event_shape) is 0:
          self._event_shapes.append(1)
        elif len(d.event_shape) is 1:
          self._event_shapes.append(d.event_shape.as_list()[0])
        else:
          raise ValueError(
              "Dimension of each distribution event_shape has to be 0 or 1")

    super(IndependentList, self).__init__(
        dtype=self._distributions[0].dtype,
        reparameterization_type=self._distributions[0].reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=self._distributions[0].allow_nan_stats,
        parameters=parameters,
        graph_parents=(
            sum([d._graph_parents for d in self._distributions],[])
        ),  # pylint: disable=protected-access
        name=name)

  @property
  def distributions(self):
    return self._distributions

  @property
  def event_shapes(self):
    return self._event_shapes

  def _batch_shape_tensor(self):
    batch_shape = self.distributions[0].batch_shape_tensor()
    return batch_shape

  def _batch_shape(self):
    batch_shape = self.distributions[0].batch_shape
    return batch_shape

  def _event_shape_tensor(self):
    event_shape = tf.convert_to_tensor([sum(self.event_shapes)], dtype=tf.float32)
    return event_shape

  def _event_shape(self):
    event_shape = tf.TensorShape([sum(self.event_shapes)])
    return event_shape

  def _sample_n(self, n, seed):
    rs = np.random.RandomState(seed=truncate_seed(seed))
    # get random seeds (have to be Python ints)
    seeds = rs.randint(low=0,high=10000, size=len(self.distributions))
    samples = [tf.reshape(d.sample(n, seed=seed),[n,-1])
               for s,d in zip(seeds,self.distributions)]
    return tf.concat(samples, axis=-1)

  def _log_prob(self, x):
    # beware caching in bijector composed distributions
    splitted_x = tf.split(x, self.event_shapes, axis=-1)
    log_probs = [tf.reshape(d.log_prob(x_i),
                   tf.concat([tf.shape(x_i)[:-1], [-1]], axis=-1))
                   for x_i, d in zip(splitted_x, self.distributions)]
    return tf.reduce_sum(tf.stack(log_probs, axis=-1), axis=-1)
