# `tf_agents.drivers`

Helper functions, for running a `policy` within an `environment`.

## Arguments

Drivers use the base class `tf_agents.drivers.driver.Driver`, and take (mostly) three arguments:

  1. `environment`
  2. `policy`
  3. `observers`
    - this is an array of (optional) callback functions, which will take a `trajectory` as the argument

## Types

Currently there are three `Driver` subclasses:

  - `DynamicEpisodeDriver`
    - stops after certain number of episodes
  - `DynamicStepDriver`
    - stops after certain number of steps
  - `PyDriver`
    - runs a Python `environment` (not a TF `environment`)
