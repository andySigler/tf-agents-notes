# TF-Agents Notes

There's no guide for TF-Agents, other than some notebooks on their repo. So, here's my attempt at making something for myself to document and learn from.

I anticipate the TF-Agents team will have some documentation in the near future, so this is mainly for me as I begin to try and figure out how to use this while teaching myself RL.

  - [Concepts](#Concepts)
    - [Interactions](#Interactions)
    - [Agents](#Agents)
    - [Data Types](#data-types)
  - [Modules](#Modules)
    - [tf_agents.specs](#tf_agentsspecs)
    - [tf_agents.trajectories](#tf_agentstrajectories)
    - [tf_agents.replay_buffers](#tf_agentsreplay_buffers)
    - [tf_agents.environments](#tf_agentsenvironments)
    - [tf_agents.agents](#tf_agentsagents)
    - [tf_agents.drivers](#tf_agentsdrivers)
    - [tf_agents.policies](#tf_agentspolicies)
    - [tf_agents.networks](#tf_agentsnetworks)
    - [tf_agents.metrics](#tf_agentsmetrics)
    - [tf_agents.utils](#tf_agentsutils)
    - [tf_agents.distributions](#tf_agentsdistributions)


# Concepts

TF-Agents is a library which uses Tensorflow's neural networking abilities to learn RL policies. That is, it maps an `environment`'s state to an `agent`'s `action`, within that `environment`, using a neural net.

I'm not yet quite clear on where specific things fit within this system, like RL rewards and NN loss functions. That stuff hasn't been explicitly described in the TF-Agents example notebooks, since the point of their code is to hide that away...

## Interactions

  - An `environment` (traditionally made using OpenAI Gym, or using their `tf_agents.environments` options) returns a `time_step`.
    - That `time_step` can contain an array of numbers (the `observation`), the `reward` (if an action was taken), and whether or not it is the final `time_step` in an episode.

  - An `agent` (which includes a handful of configurable things inside it), will interact with this `environment` by reading its `time_step`, and deciding which `action` to take next.

  - That `action` is then sent into the `environment`, which of course then returns a brand new `time_step`.

## Agents

The `agent` can choose each `action` to either:

  1. increase the likelyhood of getting the most rewards from the `time_step` (using the `agent.policy`)
  2. or, it will choose actions to simply acquire data about the evironment (using the `agent.collect_policy`).

While collecting data from the environment, the `agent` will store three things at each step in a dataset. They are the current `time_step`, the `action` it chose, and then the resulting `time_step`. These can then be used in batches later one, to train the `agent.policy` (similar to training a traditional NN).

There are then implementation differences between the different algorithms, and many points at which the `agent` is configurable, all of which it would seem require a deeper knowledge of the tools and what you are trying to Learn.

## Data Types

### Observation

A numpy array (or tensor or nest) containing the input data of a policy network (also it is contained within the output from an environment).

### Action

A numpy array (or tensor or nest) containing the output data of a policy network (also it then fed to an environment to act upon).

### Policy State

A numpy array (or tensor or nest) containing a policy's previous state.

### Discount

A numpy array (or tensor or nest) with a range of [0, 1].

# Modules

## `tf_agents.specs`

Specs allow classes to instatiate based on their expected input/output. Specs are used to document a numpy array or tensor's data type, the shape, and (optionally) the allowed ranges of a numpy array or tensor.

There are also some helper function within the submodules that are not listed here.

### Types

  - `ArraySpec`
    - `BoundedArraySpec` specifies a min and max
  - `DistributionSpec`
  - `TensorSpec`

### Arguments

  - `shape`
  - `dtype`
  - `name` (optional)

### Methods

  - `check_array(array)`
  - `from_array(array)`
  - `from_spec(spec)`

## `tf_agents.trajectories`

This module holds the commonly used data structures throughout TF-Agents. These data structures are used to fully describe the inputs/outputs of an `environment` and `agent`.

### Step Type

Defined in `tf_agents.trajectories.time_step.StepType`, this data type is used to label a time step (the output from an `environment`) such that we know when in the episode it was created.

#### Values

These are class variables, so they are not accessible through an instance. the `StepType`, when created, is simply turned in a numpy array that will have one of the following values:

  - `StepType.FIRST` -> (0)
  - `StepType.MID` -> (1)
  - `StepType.LAST` -> (2)

### Time Step

Defined in `tf_agents.trajectories.time_step.TimeStep`, this data structure is the output from an `environment`, and is used to describe its current state as it relates to an `agent`.

#### Fields

A named tuple, returned from the `environment`. It contains the following numpy arrays (or tensor or nest):

  - `TimeStep().observation`
  - `TimeStep().reward`
  - `TimeStep().step_type`
  - `TimeStep().discount`

#### Methods

Looks like the only methods it has are to functionally check the state type it contains.

  - `TimeStep().is_first()`
  - `TimeStep().is_last()`
  - `TimeStep().is_mid()`

### Policy Step

Defined in `tf_agents.trajectories.policy_step.PolicyStep`, this data structure is returned from an `agent` (or `policy`), and is fed into an `environment` to change its state.

#### Fields

A named tuple, it contains the following numpy arrays (or tensor or nest):

  - `PolicyStep().action`
  - `PolicyStep().state` (aka policy state)
  - `PolicyStep().info`

### Trajectory

Defined in `tf_agents.trajectories.trajectories.Trajectory`, this data structure is used to define a transition from one time step to another, within an `environment`. This discription includes a combination of three things:

  - the previous `TimeStep` named tuple from the `environment`
  - the following `PolicyStep` that was taken by the `agent`
  - the resulting `TimeStep` named tuple from the `environment`

#### Fields

A named tuple, it contains the following numpy arrays (or tensor or nest):

  - `Trajectory().step_type`
  - `Trajectory().observation`
  - `Trajectory().action`
  - `Trajectory().next_step_type`
  - `Trajectory().reward`
  - `Trajectory().discount`
  - `Trajectory().policy_info`
    - Note: this does not contain the policy state, but instead "auxiliary information" (whatever that means)

#### Methods

  - `Trajectory().is_first()`
  - `Trajectory().is_mid()`
  - `Trajectory().is_last()`
  - `Trajectory().is_boundary()` (same as `.is_last()`)
  - `Trajectory().replace()`

#### Module Methods

These are helpful methods exposed by the `tf_agents.trajectories.trajectories` module:

  - `trajectory.from_transition(time_step, action_step, next_time_step)`
    - converts a single transition, during an episode, onto a `Trajectory`
  - `trajectory.from_episode(observation, action, policy_info, reward, discount=None)`
    - converts nested episode information into a single `Trajectory`
    - each argument can be nested tuples, to hold the entire episode
  - `trajectory.to_transition(trajectory, next_trajectory=None)`
    - converts a trajectory (or between it and the next one) into a transition
    - returns a tuple `(time_steps, policy_steps, next_time_steps)`
      - the rewards and discounts are zero, because those can't be deduced

## `tf_agents.replay_buffers`

## `tf_agents.environments`

Returns a `time_step`, and takes an `action` to then generate a new one.

### Arguments

  - `time_step_spec`
  - `action_spec`
  - `batch_size` (optional)

### Types

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

### Methods

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

## `tf_agents.agents`

This is where the different algorithms are kept. Each algorithm implemented in TF-Agents can be loaded here as a class, all of which use the base class `tf_agents.agents.tf_agent.TFAgent`.

### Arguments

When instatiating a `TFAgent` subclass, they all take as the first two arguments the required `environment` information:

  1. the `time_step_spec` (which can be gotten from `environment.time_step_spec()`)
  2. the `action_spec` (which can be gotten from `environment.action_spec()`)

The rest of the arguments are algorithm-specific, but they all include one or more `networks`, and one or more (optional) `optimizers` for those networks.

### Attributes

Shared attributes among all `TFAgent` instances are:

  - `TFAgent.time_step_spec`
  - `TFAgent.action_spec`
  - `TFAgent.policy`
  - `TFAgent.collect_policy`
  - `TFAgent.collect_data_spec`
    - used when created a `replay_buffer`
  - `TFAgent.train_sequence_length`
    - used when converting a `replay_buffer` into a TF `dataset`

### Types

(needs a brief summary for each)

  - `tf_agents.agents.DdpgAgent`
  - `tf_agents.agents.DqnAgent`
  - `tf_agents.agents.PPOAgent`
  - `tf_agents.agents.ReinforceAgent`
  - `tf_agents.agents.SacAgent`
  - `tf_agents.agents.Td3Agent`

## `tf_agents.drivers`

Helper functions, for running a `policy` within an `environment`.

### Arguments

Drivers use the base class `tf_agents.drivers.driver.Driver`, and take (mostly) three arguments:

  1. `environment`
  2. `policy`
  3. `observers`
    - this is an array of (optional) callback functions, which will take a `trajectory` as the argument

### Types

Currently there are three `Driver` subclasses:

  - `DynamicEpisodeDriver`
    - stops after certain number of episodes
  - `DynamicStepDriver`
    - stops after certain number of steps
  - `PyDriver`
    - runs a Python `environment` (not a TF `environment`)

## `tf_agents.policies`

## `tf_agents.networks`

## `tf_agents.metrics`

## `tf_agents.utils`

## `tf_agents.distributions`

(I haven't touched this yet)

Some distributions:

  - `tf_agents.distributions.masked`
  - `tf_agents.distributions.shifted_categorical`
  - `tf_agents.distributions.tanh_bijector_stable`
