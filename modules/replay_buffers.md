# `tf_agents.replay_buffers`

A replay buffer is used for storing `Trajectories`, which occur during collection from an `environment`. Then, once collected, a `policy` can be trained, similar to a traditional NN.

Base class is `tf_agents.replay_buffers.replay_buffer.ReplayBuffer`

There is both a Python and a TF version of the replay buffer:

  - `tf_agents.replay_buffers.py_uniform_replay_buffer.PyUniformReplayBuffer`
  - `tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer`

## Arguments

The Python version seems to only have two arguments:

  - `data_spec`
    - An `ArraySpec` or a list/tuple/nest of `ArraySpecs` describing a single item that can be stored in this buffer
  - `capacity`
    - The maximum number of items that can be stored in the buffer.

The TF version has many more arguments:

  - `data_spec`
  - `batch-size`
    - Batch dimension of tensors when adding to buffer
  - `max_length`
  - `scope`
  - `device`
  - `dataset_drop_remainder`
  - `dataset_window_shift`

## Attributes

  - `ReplayBuffer().capacity`
  - `ReplayBuffer().data_spec`

## Methods

  - `ReplayBuffer().add_batch(trajectory)`
  - `ReplayBuffer().as_dataset(num_steps=2)`
    - why do all the examples set `num_steps=2` ??
  - `ReplayBuffer().clear()`
  - `ReplayBuffer().gather_all()`
  - `ReplayBuffer().get_next()`
