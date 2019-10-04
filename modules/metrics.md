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

# `tf_agents.metrics`

There are both Python and TF metrics, plus metrics that are specificaly for `trajectories`, or "steps".

  - `tf_agents.metrics.py_metric.PyMetric`
  - `tf_agents.metrics.py_metric.PyStepMetric`
  - `tf_agents.metrics.tf_metric.TFStepMetric`

You can also wrap a Python metric into a TF metric, using:

  - `tf_agents.metrics.tf_py_metric.TFPyMetric`

## Common Classes

These methods are available in both the `py_metric` and `tf_metric` submodules:

  - `AverageEpisodeLengthMetric`
  - `AverageReturnMetric`
  - `EnvironmentStep`
  - `NumberOfEpisodes`

## Python Classes

  - `NumpyDeque`
  - `CounterMetric`
  - `StreamingMetric`

## TF Classes

  - `TFDeque`
