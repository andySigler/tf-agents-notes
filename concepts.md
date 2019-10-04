# Concepts

## Interactions

  - An `environment` (traditionally made using OpenAI Gym, or using their `tf_agents.environments` options) returns a `time_step`.
    - That `time_step` can contain an array of numbers (the `observation`), the `reward` (if an action was taken), and whether or not it is the final `time_step` in an episode.

  - An `agent` (which includes a handful of configurable things inside it), will interact with this `environment` by reading its `time_step`, and deciding which `action` to take next.

  - That `action` is then sent into the `environment`, which of course then returns a brand new `time_step`.

The `agent` can choose each `action` to either:

  1. increase the likelyhood of getting the most rewards from the `time_step` (using the `agent.policy`)
  2. or, it will choose actions to simply acquire data about the evironment (using the `agent.collect_policy`).

While collecting data from the environment, the `agent` will store three things at each step in a dataset. They are the current `time_step`, the `action` it chose, and then the resulting `time_step`. These can then be used in batches later one, to train the `agent.policy` (similar to training a traditional NN).

There are then implementation differences between the different algorithms, and many points at which the `agent` is configurable, all of which it would seem require a deeper knowledge of the tools and what you are trying to Learn.

## Data Types

There seem to be found main data types used in the library, each listed below.

These basic data types are then organized into named tuples throughout TF-Agents, to facilitate the interactions described above.

In addition, `specs` are used throughout the code to describe these data types while instantiating a class.

### Observation

A numpy array (or tensor or nest) containing the input data of a policy network (also it is contained within the output from an environment).

### Action

A numpy array (or tensor or nest) containing the output data of a policy network (also it then fed to an environment to act upon).

### Policy State

A numpy array (or tensor or nest) containing a policy's previous state.

### Discount

A numpy array (or tensor or nest) with a range of [0, 1].
