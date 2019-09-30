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

TF-Agents is a library which uses Tensorflow's neural networking abilities to learn RL policies. That is, it maps an __`environment`__'s state to an __`agent`__'s __`action`__, within that __`environment`__, using a neural net.

I'm not yet quite clear on where specific things fit within this system, like RL rewards and NN loss functions. That stuff hasn't been explicitly described in the TF-Agents example notebooks, since the point of their code is to hide that away...

## Interactions

  - An __`environment`__ (traditionally made using OpenAI Gym, or using their __`tf_agents.environments`__ options) returns a __`time_step`__.
    - That __`time_step`__ can contain an array of numbers (the __`observation`__), the __`reward`__ (if an action was taken), and whether or not it is the final __`time_step`__ in an episode.

  - An __`agent`__ (which includes a handful of configurable things inside it), will interact with this __`environment`__ by reading its __`time_step`__, and deciding which __`action`__ to take next.

  - That __`action`__ is then sent into the __`environment`__, which of course then returns a brand new __`time_step`__.

## Agents

The __`agent`__ can choose each __`action`__ to either:

  1. increase the likelyhood of getting the most rewards from the __`time_step`__ (using the __`agent.policy`__)
  2. or, it will choose actions to simply acquire data about the evironment (using the __`agent.collect_policy`__).

While collecting data from the environment, the __`agent`__ will store three things at each step in a dataset. They are the current __`time_step`__, the __`action`__ it chose, and then the resulting __`time_step`__. These can then be used in batches later one, to train the __`agent.policy`__ (similar to training a traditional NN).

There are then implementation differences between the different algorithms, and many points at which the __`agent`__ is configurable, all of which it would seem require a deeper knowledge of the tools and what you are trying to Learn.

## Data Types

### Action

Action namped tuples

### Time Step

Time step namped tuples

# Modules

## `tf_agents.specs`

There are also some helper function within the submodules that are not listed here.

### Types

  - __`ArraySpec`__
    - __`BoundedArraySpec`__ specifies a min and max
  - __`DistributionSpec`__
  - __`TensorSpec`__

### Arguments

  - `shape`
  - `dtype`
  - `name` (optional)

### Methods

  - `check_array(array)`
  - `from_array(array)`
  - `from_spec(spec)`

## `tf_agents.trajectories`

## `tf_agents.replay_buffers`

## `tf_agents.environments`

Returns a __`time_step`__, and takes an __`action`__ to then generate a new one.

### Arguments

  - __`time_step_spec`__
  - __`action_spec`__
  - __`batch_size`__ (optional)

### Types

There are two types of environments in TF-Agents, one for a pure Python environment, and another separate type for a TF environment.

For Python, the base class is __`tf_agents.environments.py_environment.PyEnvironment`__.

  - __`PyEnvironment`__
  - __`RandomPyEnvironment`__
  - __`BatchedPyEnvironment`__
  - __`ParallelPyEnvironment`__

For TF, it's __`tf_agents.environments.tf_environment.TFEnvironment`__.

  - __`TFEnvironment`__
  - __`RandomTFEnvironment`__

You can "wrap" a Python __`environment`__ as a TF, using __`tf_agents.environments.tf_py_environment.TFPyEnvironment`__.

There are also pre-made environments, which can be loaded from __`tf_agents.environments.suite_gym.load(name)`__. The argument __`name`__ is a string that is included in the OpenAI Gym registry.

  - [Available environments from OpenAI](http://gym.openai.com/envs/#classic_control)

### Methods

Methods that return a __`time_step`__:

  - __`env.reset()`__
  - __`env.step(action)`__
  - __`env.current_time_step()`__

Methods that return a __`spec`__:

  - __`env.observation_spec()`__
  - __`env.action_spec()`__
  - __`env.time_step_spec()`__

Other:

  - __`env.render()`__

## `tf_agents.agents`

This is where the different algorithms are kept. Each algorithm implemented in TF-Agents can be loaded here as a class, all of which use the base class __`tf_agents.agents.tf_agent.TFAgent`__.

### Arguments

When instatiating a __`TFAgent`__ subclass, they all take as the first two arguments the required __`environment`__ information:

  1. the __`time_step_spec`__ (which can be gotten from __`environment.time_step_spec()`__)
  2. the __`action_spec`__ (which can be gotten from __`environment.action_spec()`__)

The rest of the arguments are algorithm-specific, but they all include one or more __`networks`__, and one or more (optional) __`optimizers`__ for those networks.

### Attributes

Shared attributes among all __`TFAgent`__ instances are:

  - __`TFAgent.time_step_spec`__
  - __`TFAgent.action_spec`__
  - __`TFAgent.policy`__
  - __`TFAgent.collect_policy`__
  - __`TFAgent.collect_data_spec`__
    - used when created a __`replay_buffer`__
  - __`TFAgent.train_sequence_length`__
    - used when converting a __`replay_buffer`__ into a TF __`dataset`__

### Types

(needs a brief summary for each)

  - __`tf_agents.agents.DdpgAgent`__
  - __`tf_agents.agents.DqnAgent`__
  - __`tf_agents.agents.PPOAgent`__
  - __`tf_agents.agents.ReinforceAgent`__
  - __`tf_agents.agents.SacAgent`__
  - __`tf_agents.agents.Td3Agent`__

## `tf_agents.drivers`

Helper functions, for running a __`policy`__ within an __`environment`__.

### Arguments

Drivers use the base class __`tf_agents.drivers.driver.Driver`__, and take (mostly) three arguments:

  1. __`environment`__
  2. __`policy`__
  3. __`observers`__
    - this is an array of (optional) callback functions, which will take a __`trajectory`__ as the argument

### Types

Currently there are three __`Driver`__ subclasses:

  - __`DynamicEpisodeDriver`__
    - stops after certain number of episodes
  - __`DynamicStepDriver`__
    - stops after certain number of steps
  - __`PyDriver`__
    - runs a Python __`environment`__ (not a TF __`environment`__)

## `tf_agents.policies`

## `tf_agents.networks`

## `tf_agents.metrics`

## `tf_agents.utils`

## `tf_agents.distributions`

(I haven't touched this yet)

Some distributions:

  - __`tf_agents.distributions.masked`__
  - __`tf_agents.distributions.shifted_categorical`__
  - __`tf_agents.distributions.tanh_bijector_stable`__
