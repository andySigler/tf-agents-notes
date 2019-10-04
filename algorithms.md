# Algorithms

## Notation

- `θ`: the parameters of a policy; the weights
- `πθ`: a stochastic policy, paramaterized by some weights
- `μθ`: a deterministic policy, paramaterized by some weights
- `α`: learning rate; step size
- `γ`: discount factor; uncertainty of future rewards
- `Gt`: Return; discounted future reward
- `∇`: gradient; derivative
- `ln`: natural log

## REINFORCE

- On-policy
- Policy-gradient based
- Monte Carlo methods (episodic)

The update methods is:

```
- Initialize the policy parameter θ at random.
- Generate one trajectory on policy πθ: S1,A1,R2,S2,A2,…,ST
- For t=1,2,…,T:
    - Estimate the the return Gt
    - Update policy parameters: θ ← θ + α (γ * Gt * ∇θ * ln[πθ(At|St)])
```

It can be high variance. A common methods to reduce variance is to subtract a baseline value from the return `Gt`

## Actor-Critic

This describes a "vanilla" version of actor-critic

- On-policy
- Policy-gradient based

To reduce the variance of policy-gradient methods, we can also learn the value function, in addition to the policy model (like in `REINFORCE`).

The __critic__ updates the value function parameters, `w`

- this could be either state-value `Vw(s)` or action-value `Qw(a|s)`, depending on the algorithm

the __actor__ updates the policy parameter `θ` for `πθ(a|s)`, in the direction suggested by the __critic__

The update method is:

```
- Initialize s, θ, w at random; sample a ∼ πθ(a|s)
- For t=1…T
    - Sample reward rt ∼ R(s,a), and next state s′ ∼ P(s′|s,a)
    - Then sample the next action a′ ∼ πθ(a′|s′)
    - Update the policy parameters: θ ← θ + αθ * Qw(s,a) * ∇θ * ln[πθ(a|s)]
    - Compute the correction (TD error) for action-value at time t:
        - δt = rt + γ * Qw(s′,a′) − Qw(s,a)
    - Then use it to update the parameters of action-value function:
        - w ← w + αw * δt * ∇w * Qw(s,a)
    - Update a ← a′` and `s ← s′.
```

## Off-Policy Policy Gradient

Both `REINFORCE` and the "vanilla" version of actor-critic method are on-policy: training samples are collected according to the target policy — the very same policy that we try to optimize for. Off-policy methods, however, result in additional advantages:

- The off-policy approach does not require full trajectories and can reuse any past episodes (“experience replay”) for much better sample efficiency.
- The sample collection follows a behavior policy different from the target policy, bringing better exploration.

## PPO

- on-policy

## DQN

- off-policy
- only works with discrete action space

Instead of storing Q values for each state, we can estimate the Q value using a NN, given weight parameters `θ`.

The downside is that this can be very unstable and/or can diverge.

`DQN` aims to stablize and improve by doing two things:

- Experience Replay
    - samples are drawn at random during training, which can smooth over the data distribution
- Periodically Updated target
    - Q is optimized towards target values that are only periodically updated. The Q network is cloned and kept frozen as the optimization target every C steps (C is a hyperparameter). This modification makes the training more stable as it overcomes the short-term oscillations.

## DDQN

## DDPG

- model-free
- actor-critic
- off-policy
- continuous action space
- deterministic (outputs single action)
- combines `DPG` and `DQN`

## TD3

Applies a couple of tricks on `DDPG` to prevent the overestimation of the value function.

- Clipped Double Q-learning
- Delayed update of Target and Policy Networks
- Target Policy Smoothing

## SAC

- off-policy
- actor-critic

We expect to learn a policy that acts as randomly as possible while it is still able to succeed at the task. The policy is trained with the objective to maximize the expected return and the entropy at the same time.


