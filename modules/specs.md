# `tf_agents.specs`

Specs allow classes to instatiate based on their expected input/output. Specs are used to document a numpy array or tensor's data type, the shape, and (optionally) the allowed ranges of a numpy array or tensor.

There are also some helper function within the submodules that are not listed here.

## Types

  - `ArraySpec`
    - `BoundedArraySpec` specifies a min and max
  - `DistributionSpec`
  - `TensorSpec`

## Arguments

  - `shape`
  - `dtype`
  - `name` (optional)

## Methods

  - `check_array(array)`
  - `from_array(array)`
  - `from_spec(spec)`
