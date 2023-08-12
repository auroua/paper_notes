# Reinforcement Learning: An Introduction

## Chapter 1 Introduction

## Chapter 2 Multi-armed Bandits

## Chapter 3 Finite Markov Decision Processes

## Chapter 4 Dynamic Programming

## Chapter 5 Monte Calo Methods

## Chapter 6 Temporal-Difference Learning

## Chapter 7 n-step Boostrapping

### n-step prediction

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+n} + \gamma ^{n} V_\pi(S_{t+n})
$$

$$
V_\pi(S_t) = V_\pi(S_t) + \alpha (G_{t:t+n} - V_\pi(S_t))
$$

### n-step SARSA

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+n} + \gamma ^{n} Q_\pi(S_{t+n}, A_{t+n})
$$

$$
Q_\pi(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha (G_{t:t+n} - Q_\pi(S_t, A_t))
$$

### n-step Q-Learning

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+n} + \gamma ^{n} \max _a Q_\pi(S_{t+n}, a)
$$

$$
Q_\pi(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha (G_{t:t+n} - Q_\pi(S_t, A_t))
$$

### n-step off-policy learning

#### Prediction

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+n} + \gamma ^{n} V_\pi(S_{t+n})
$$

$$
V_\pi(S_t) = V_\pi(S_t) + \alpha \rho_{t:t+n-1} (G_{t:t+n} - V_\pi(S_t))
$$

#### Control

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+n} + \gamma ^{n} Q_\pi(S_{t+n}, A_{t+n})
$$

$$
Q_\pi(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha \rho_{t+1:t+n} (G_{t:t+n} - Q_\pi(S_t, A_t))
$$


## Chapter 8 Planning and Learning with Tabular Methods

## Chapter 9 On-policy Prediction with Approximation

## Chapter 10 On-policy Control with Approximation

### Episodic Semi-gradient SARSA

$$
\delta = R_{t+1} + Q_\pi(S_{t+1}, A_{t+1}; w) - Q_\pi(S_t, A_t; w)
$$

$$
w_t = w_t + \alpha\delta \nabla_w Q_\pi(S_t, A_t; w)
$$

### Episodic n-step Semi-gradient SARSA

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+1} + \gamma ^nQ_\pi(S_{t+n}, A_{t+n}; w)
$$

$$
\delta = G_{t:t+n} - Q(S_t, A_t; w)
$$

$$
w_t = w_t + \alpha \delta \nabla_w Q_\pi(S_t, A_t; w)
$$

### Differential Semi-gradient SARSA (using average reward for continuing problem)

$$
G_{t:t+1} = R_{t+1} - \overline R + Q_\pi(S_{t+1}, A_{t+1}; w) - Q(S_t, A_t; w)
$$

$$
w_t = w_t + \alpha (G_{t:t+1} - Q_\pi(S_t, A_t; w)) \nabla_wQ(S_t, A_t; w)
$$

### Differential Semi-gradient n-step SARSA (using average reward for continuing problem)

$$
G_{t:t+n} = R_{t+1} - \overline R + R_{t+2} - \overline R + R_{t+3} - \overline R + \dots + Q_\pi(S_{t+n}; A_{t+n}; w)
$$

$$
w_t = w_t + \alpha(G_{t:t+n} - Q_\pi(S_t, A_t; w)) \nabla_w Q(S_t, A_t; w)
$$

## Chpater 12 Eligibility Traces

### $\lambda$-return

This section introduction a new kind of update target $G_t^\lambda$.

#### Prediction Problem

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n v_\pi(S_{t+1;w})
$$

$$
G_t^\lambda = (1-\lambda) \sum_{n=1}^\infty \lambda ^ n G_{t:t+n}
$$

**off-line update**
$$
w_t = w_t + \alpha \times (G_t^\lambda - v_\pi(S_t; w)) \nabla_w v_\pi(S_t; w)
$$

#### Control Problem

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma ^n Q_\pi(S_{t+n}, A_{t+n}; w)
$$

$$
G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}G_{t:t+n}
$$

**off-line update**
$$
w_t = w_t + \alpha (G_t^\lambda - Q(S_t, A_t; w)) \nabla_w Q(S_t, A_t; w)
$$

### TD($\lambda$)

#### Prediction Problem

An interaction trajectory with $S_t$, $A_t$, $R_{t+1}$, $S_{t+1}$, and $A_{t+1}$.

$$
z_{-1} = 0
$$

$$
z_{t} = \gamma \lambda z_{t-1} + \nabla_w V_\pi (S_t; w)
$$

$$
\delta = R_{t+1} + \gamma V_\pi(S_{t+1}) - V_\pi(S_t)
$$

$$
w_t = w_t + \alpha \delta z_{t}
$$

#### Control Problem SARSA($\lambda$)

An interaction trajectory with $S_t$, $A_t$, $R_{t+1}$, $S_{t+1}$, and $A_{t+1}$.

$$
z_{-1} = 0
$$

$$
z_t = \gamma \lambda z_{t-1} + \nabla_w Q(S_{t+1}, A_{t+1}; w)
$$

$$
\delta = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}; w) - Q(S_t, A_t; w)
$$

$$
w_t = w_t + \alpha \delta z_t
$$

## Chapter 13 Policy Gradient Methods