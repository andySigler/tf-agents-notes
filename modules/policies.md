# `tf_agents.policies`

A call to `policy.action(time_step, policy_state=())` will return a `PolicyStep` named tuple, on which an environment can act upon.

There are both Python and TF policies:

  - `tf_agents.policies.py_policy.PyPolicy`
  - `tf_agents.policies.tf_policy.TFPolicy`

You can also wrap a Python policy into a TF policy, using:

  - `tf_agents.policies.tf_py_policy.TFPyPolicy`

## Attributes

  - `policy.action_spec`
  - `policy.info_spec`
  - `policy.policy_state_spec`
  - `policy.policy_step_spec`
  - `policy.time_step_spec`
  - `policy.trajectory_spec`

## Methods

  - `policy.action(time_step, policy_state=())`
  - `policy.distribution(time_step, policy_state=())`

## Basic Types

  - `tf_agents.policies.fixed_policy`
  - `tf_agents.policies.random_py_policy`
  - `tf_agents.policies.random_tf_policy`
  - `tf_agents.policies.scripted_py_policy`

## Simple Types

  - `tf_agents.policies.q_policy`
  - `tf_agents.policies.actor_policy`

## Wrappers

  - `tf_agents.policies.greedy_policy`
  - `tf_agents.policies.epsilon_greedy_policy`
  - `tf_agents.policies.gaussian_policy`
  - `tf_agents.policies.ou_noise_policy`
  - `tf_agents.policies.boltzmann_policy`

## Utils

  - `tf_agents.policies.policy_saver`
