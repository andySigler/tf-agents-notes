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
