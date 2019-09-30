# TF-Agents Notes

There's no guide for TF-Agents, other than some notebooks on their repo. So, here's my attempt at making something for myself to document and learn from.

I anticipate the TF-Agents team will have some documentation in the near future, so this is mainly for me as I begin to try and figure out how to use this while teaching myself RL.

## Overview

TF-Agents is a library which uses Tensorflow's neural networking abilities to learn RL policies. That is, it maps an __`environment`__'s state to an __`agent`__'s __`action`__, within that __`environment`__, using a neural net.

I'm not yet quite clear on where specific things fit within this system, like RL rewards and NN loss functions. That stuff hasn't been explicitly described in the TF-Agents example notebooks, since the point of their code is to hide that away...

### Environment Interactions

  - An __`environment`__ (traditionally made using OpenAI Gym, or using their __`tf_agents.environments`__ options) returns a __`time_step`__.
    - That __`time_step`__ can contain an array of numbers (the __`observation`__), the __`reward`__ (if an action was taken), and whether or not it is the final __`time_step`__ in an episode.

  - An __`agent`__ (which includes a handful of configurable things inside it), will interact with this __`environment`__ by reading its __`time_step`__, and deciding which __`action`__ to take next.

  - That __`action`__ is then sent into the __`environment`__, which of course then returns a brand new __`time_step`__.

### Agents

The __`agent`__ can either choose each __`action`__ to either:

  1. increase the likelyhood of getting the most rewards from the __`time_step`__ (using the __`agent.policy`__)
  2. or, it will choose actions to simply acquire data about the evironment (using the __`agent.collect_policy`__).

While collecting data from the environment, the __`agent`__ will store three things at each step in a dataset. They are the current __`time_step`__, the __`action`__ it chose, and then the resulting __`time_step`__. These can then be used in batches later one, to train the __`agent.policy`__ (similar to training a traditional NN).

There are then implementation differences between the different algorithms, and many points at which the __`agent`__ is configurable, all of which it would seem require a deeper knowledge of the tools and what you are trying to Learn.

## Modules

### `tf_agents.agents`

This is where the different algorithms are kept. Each algorithm implemented in TF-Agents can be loaded here as a class, all of which use the base class __`tf_agents.agents.tf_agent.TFAgent`__.

#### Instantiating

When instatiating a __`TFAgent`__ subclass, they all take as the first two arguments the required __`environment`__ information:

  1. the __`time_step_spec`__ (which can be gotten from __`environment.time_step_spec()`__)
  2. the __`action_spec`__ (which can be gotten from __`environment.action_spec()`__)

The rest of the arguments are algorithm-specific, but they all include one or more __`networks`__, and one or more (optional) __`optimizers`__ for those networks.

#### Attributes

Shared attributes among all __`TFAgent`__ instances are:

  - __`TFAgent.time_step_spec`__
  - __`TFAgent.action_spec`__
  - __`TFAgent.policy`__
  - __`TFAgent.collect_policy`__
  - __`TFAgent.collect_data_spec`__
    - used when created a __`replay_buffer`__
  - __`TFAgent.train_sequence_length`__
    - used when converting a __`replay_buffer`__ into a TF `dataset`


### `tf_agents.distributions`

### `tf_agents.drivers`

### `tf_agents.environments`

### `tf_agents.metrics`

### `tf_agents.networks`

### `tf_agents.policies`

### `tf_agents.replay_buffers`

### `tf_agents.specs`

### `tf_agents.trajectories`

### `tf_agents.utils`
