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

# `tf_agents.agents`

This is where the different algorithms are kept. Each algorithm implemented in TF-Agents can be loaded here as a class, all of which use the base class `tf_agents.agents.tf_agent.TFAgent`.

## Arguments

When instatiating a `TFAgent` subclass, they all take as the first two arguments the required `environment` information:

  1. the `time_step_spec` (which can be gotten from `environment.time_step_spec()`)
  2. the `action_spec` (which can be gotten from `environment.action_spec()`)

The rest of the arguments are algorithm-specific, but they all include one or more `networks`, and one or more (optional) `optimizers` for those networks.

## Attributes

Shared attributes among all `TFAgent` instances are:

  - `TFAgent.time_step_spec`
  - `TFAgent.action_spec`
  - `TFAgent.policy`
  - `TFAgent.collect_policy`
  - `TFAgent.collect_data_spec`
    - used when created a `replay_buffer`
  - `TFAgent.train_sequence_length`
    - used when converting a `replay_buffer` into a TF `dataset`

## Types

(needs a brief summary for each)

### On-Policy (Policy Gradient)

  - `tf_agents.agents.ReinforceAgent`
    - `actor_network`
    - `value_network` (optional)
  - `tf_agents.agents.PPOAgent`
    - `actor_net`
    - `value_net`

### Off-Policy (Value Function)

  - `tf_agents.agents.DqnAgent`
    - `q_network`
  - `tf_agents.agents.dqn.dqn_agent.DdqnAgent`
    - `q_network`

### Hybrid (Actor-Critic)

  - `tf_agents.agents.DdpgAgent`
    - `actor_network`
    - `critic_network`
  - `tf_agents.agents.Td3Agent`
    - `actor_network`
    - `critic_network`
  - `tf_agents.agents.SacAgent`
    - `actor_network`
    - `critic_network`


