# TF-Agents Notes

There's no guide for TF-Agents, other than some notebooks on their repo. So, here's my attempt at making something for myself to document and learn from.

I anticipate the TF-Agents team will have some documentation in the near future, so this is mainly for me as I begin to try and figure out how to use this while teaching myself RL.

## Overview

TF-Agents is a library which uses Tensorflow's neural networking abilities to learn RL policies. That is, it maps an `environment`'s state to an `agent`'s `action`, within that `environment`, using a neural net.

I'm not yet quite clear on where specific things fit within this system, like RL rewards and NN loss functions. That stuff hasn't been explicitly described in the TF-Agents example notebooks, since the point of their code is to hide that away...

The main interaction:

  1. n `environment` (traditionally made using OpenAI Gym, or using their `tf_agents.environments` options) returns a `time_step`.
    - That `time_step` can contain an array of numbers (the `observation`), the `reward` (if an action was taken), and whether or not it is the final `time_step` in an episode.
  2. An `agent` (which includes a handful of configurable things inside it), will interact with this `environment` by reading its `time_step`, and deciding which `action` to take next.

  3. That `action` is then sent into the `environment`, which of course then returns a brand new `time_step`.

The `agent` can either choose `action`s to either:

  1. increase the likelyhood of getting the most rewards from the `time_step` (using the `agent.policy`)
  2. or, it will choose actions to simply acquire data about the evironment (using the `agent.collect_policy`).

While collecting data from the environment, the `agent` will store three things at each step in a dataset. They are the current `time_step`, the `action` it chose, and then the resulting `time_step`. These can then be used in batches later one, to train the `agent.policy` (similar to training a traditional NN).

That seems to be essentially it. There are then implementation differences between the different algorithms, and many points at which the `agent` is configurable, all of which it would seem require a deeper knowledge of the tools and what you are trying to Learn.

## Modules

### `tf_agents.agents`

This is where the different algorithms are kept. Each algorithm implemented in TF-Agents can be loaded here as a class, all of which use the base class `tf_agents.agents.tf_agent.TFAgent`.

When instatiating a `TFAgent` subclass, they all take as the first two arguments the required `environment` information:

  1. the `time_step_spec` (which can be gotten from `environment.time_step_spec()`)
  2. the `action_spec` (which can be gotten from `environment.action_spec()`)

The rest of the arguments are algorithm-specific, but they all include one or more `networks`, and one or more (optional) `optimizers` for those networks.

Shared attributes among all `TFAgent` instances are:

  - `TFAgent.time_step_spec`
  - `TFAgent.action_spec`
  - `TFAgent.policy`
  - `TFAgent.collect_policy`
  - `TFAgent.collect_data_spec`
  - `TFAgent.train_sequence_length`


### `tf_agents.distributions`

### `tf_agents.drivers`

### `tf_agents.environments`

### `tf_agents.metrics`

### `tf_agents.networks`

### `tf_agents.policies`

### `tf_agents.replay_buffers`

### `tf_agents.specs`

### `tf_agents.trajectories`

### `tf_agents.utils`
