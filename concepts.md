# Contents

  - [Concepts](./concepts.md)
    - [Interactions](./concepts.md#Interactions)
    - [Agents](./concepts.md#Agents)
    - [Data Types](./concepts.md#data-types)
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
