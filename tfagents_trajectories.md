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

# `tf_agents.trajectories`

This module holds the commonly used data structures throughout TF-Agents. These data structures are used to fully describe the inputs/outputs of an `environment` and `agent`.

## Step Type

Defined in `tf_agents.trajectories.time_step.StepType`, this data type is used to label a time step (the output from an `environment`) such that we know when in the episode it was created.

### Values

These are class variables, so they are not accessible through an instance. the `StepType`, when created, is simply turned in a numpy array that will have one of the following values:

  - `StepType.FIRST` -> (0)
  - `StepType.MID` -> (1)
  - `StepType.LAST` -> (2)

## Time Step

Defined in `tf_agents.trajectories.time_step.TimeStep`, this data structure is the output from an `environment`, and is used to describe its current state as it relates to an `agent`.

### Fields

A named tuple, returned from the `environment`. It contains the following numpy arrays (or tensor or nest):

  - `TimeStep().observation`
  - `TimeStep().reward`
  - `TimeStep().step_type`
  - `TimeStep().discount`

### Methods

Looks like the only methods it has are to functionally check the state type it contains.

  - `TimeStep().is_first()`
  - `TimeStep().is_last()`
  - `TimeStep().is_mid()`

## Policy Step

Defined in `tf_agents.trajectories.policy_step.PolicyStep`, this data structure is returned from an `agent` (or `policy`), and is fed into an `environment` to change its state.

### Fields

A named tuple, it contains the following numpy arrays (or tensor or nest):

  - `PolicyStep().action`
  - `PolicyStep().state` (aka policy state)
  - `PolicyStep().info`

## Trajectory

Defined in `tf_agents.trajectories.trajectories.Trajectory`, this data structure is used to define a transition from one time step to another, within an `environment`. This discription includes a combination of three things:

  - the previous `TimeStep` named tuple from the `environment`
  - the following `PolicyStep` that was taken by the `agent`
  - the resulting `TimeStep` named tuple from the `environment`

### Fields

A named tuple, it contains the following numpy arrays (or tensor or nest):

  - `Trajectory().step_type`
  - `Trajectory().observation`
  - `Trajectory().action`
  - `Trajectory().next_step_type`
  - `Trajectory().reward`
  - `Trajectory().discount`
  - `Trajectory().policy_info`
    - Note: this does not contain the policy state, but instead "auxiliary information" (whatever that means)

### Methods

  - `Trajectory().is_first()`
  - `Trajectory().is_mid()`
  - `Trajectory().is_last()`
  - `Trajectory().is_boundary()` (same as `.is_last()`)
  - `Trajectory().replace()`

### Module Methods

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
