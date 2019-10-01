# TF-Agents Notes

  - [Concepts](./concepts.md)
    - [Interactions](./concepts.md#Interactions)
    - [Agents](./concepts.md#Agents)
    - [Data Types](./concepts.md#data-types)
  - [Modules](./modules.md)
    - [tf_agents.specs](./tfagents_specs.md)
    - [tf_agents.trajectories](./tfagents_trajectories.md)
    - [tf_agents.replay_buffers](./tfagents_replay_buffers.md)
    - [tf_agents.environments](./tfagents_environments.md)
    - [tf_agents.agents](./tfagents_agents.md)
    - [tf_agents.drivers](./tfagents_drivers.md)
    - [tf_agents.policies](./tfagents_policies.md)
    - [tf_agents.networks](./tfagents_networks.md)
    - [tf_agents.metrics](./tfagents_metrics.md)
    - [tf_agents.utils](./tfagents_utils.md)
    - [tf_agents.distributions](./tfagents_distributions.md)

# `tf_agents.environments`

Returns a `time_step`, and takes an `action` to then generate a new one.

## Arguments

  - `time_step_spec`
  - `action_spec`
  - `batch_size` (optional)

## Types

There are two types of environments in TF-Agents, one for a pure Python environment, and another separate type for a TF environment.

For Python, the base class is `tf_agents.environments.py_environment.PyEnvironment`.

  - `PyEnvironment`
  - `RandomPyEnvironment`
  - `BatchedPyEnvironment`
  - `ParallelPyEnvironment`

For TF, it's `tf_agents.environments.tf_environment.TFEnvironment`.

  - `TFEnvironment`
  - `RandomTFEnvironment`

You can "wrap" a Python `environment` as a TF, using `tf_agents.environments.tf_py_environment.TFPyEnvironment`.

There are also pre-made environments, which can be loaded from `tf_agents.environments.suite_gym.load(name)`. The argument `name` is a string that is included in the OpenAI Gym registry.

  - [Available environments from OpenAI](http://gym.openai.com/envs/#classic_control)

## Methods

Methods that return a `time_step`:

  - `env.reset()`
  - `env.step(action)`
  - `env.current_time_step()`

Methods that return a `spec`:

  - `env.observation_spec()`
  - `env.action_spec()`
  - `env.time_step_spec()`

Other:

  - `env.render()`
