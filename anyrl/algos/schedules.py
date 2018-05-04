"""
Schedules for learning rates and other hyper-parameters.
"""

from abc import ABC, abstractmethod

import tensorflow as tf


class TFSchedule(ABC):
    """
    An abstract scheduled value for TensorFlow graphs.

    This can be used for annealed learning rates, among
    other things.

    Attributes:
      value: a Tensor with the current scheduled value.
      time: a Tensor indicating the current time.
      add_ph: a placeholder indicating how much time to
        add to the time counter.
      add_op: an Op which adds add_ph to time. It will
        never execute before self.value in the graph.
    """

    def __init__(self, dtype=tf.float32):
        time = tf.Variable(0, dtype=dtype, name='ScheduleCounter', trainable=False)
        self.value = self.compute_schedule(time)
        self.time = time
        self.add_ph = tf.placeholder(dtype)
        with tf.control_dependencies([self.value]):
            self.add_op = tf.assign_add(self.time, self.add_ph)

    def add_time(self, sess, amount):
        """
        Add the amount of time to the counter.

        Args:
          sess: the TensorFlow session.
          amount: the time to add.
        """
        sess.run(self.add_op, {self.add_ph: amount})

    @abstractmethod
    def compute_schedule(self, cur_time):
        """
        Compute the schedule value given the timestamp
        stored in cur_time.
        """
        pass


class LinearTFSchedule(TFSchedule):
    """
    A schedule that linearly interpolates between a start
    and an end value.
    """

    def __init__(self, duration=1.0, start_value=1.0, end_value=0.0, dtype=tf.float64):
        """
        Create a linear schedule.

        Args:
          duration: the timestamp at which the value
            should arrive at end_value.
          start_value: the initial value.
          end_value: the final value.
        """
        self._duration = float(duration)
        self._start_value = float(start_value)
        self._end_value = float(end_value)
        super(LinearTFSchedule, self).__init__(dtype=dtype)

    def compute_schedule(self, cur_time):
        frac_done = tf.clip_by_value(cur_time/self._duration, 0, 1)
        return (1-frac_done)*self._start_value + frac_done*self._end_value


class TFScheduleValue:
    """
    A wrapper around a TFSchedule that supports conversion
    to float via the float() built-in.
    """

    def __init__(self, sess, schedule):
        self.session = sess
        self.schedule = schedule

    def __float__(self):
        return self.session.run(self.schedule.value)
