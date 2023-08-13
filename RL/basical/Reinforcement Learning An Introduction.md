# Reinforcement Learning: An Introduction

## Chapter 4 Dynamic Programming

## Chapter 5 Monte Carlo Methods

## Chapter 6 Temporal-Difference Learning

## Chapter 7 n-step Boostrapping

>In many applications one wants to be able to update the action very fast to take into account anything that has changed, but bootstrapping works best if it is over a length of time in which a significant and recognizable state change has occurred.

### n-step prediction

>One kind of intermediate method, then, would perform an update based on an intermediate number of rewards: more than one, but less than all of them until termination. For example, a two-step update would be based on the first two rewards and the estimated value of the state two steps later.

![n-step update](../../figures/RL/rl_chp7_fig1.png)

> Methods in which the temporal difference extends over n steps are called n-step TD methods.

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+n} + \gamma ^{n} V_\pi(S_{t+n})
$$

$$
V_\pi(S_t) = V_\pi(S_t) + \alpha (G_{t:t+n} - V_\pi(S_t))
$$

> All $n$-step returns can be considered approximations to the full return, truncated after $n$ steps and then corrected for the remaining missing terms by $V(S_{t+n})$.

> No real algorithm can use the $n$-step return until after it has seen $R_{t+n}$ and computed $V_\pi(S_{t+n})$.

> The $n$-step TD methods thus form a family of sound methods, with one-step TD methods and Monte Carlo methods as extreme members.

![n-step prediction](../../figures/RL/rl_chp7_fig2.png)


### n-step SARSA

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^ {n-1} R_{t+n} + \gamma ^{n} Q_\pi(S_{t+n}, A_{t+n})
$$

$$
Q_\pi(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha (G_{t:t+n} - Q_\pi(S_t, A_t))
$$

![n-step prediction](../../figures/RL/rl_chp7_fig3.png)

![n-step prediction](../../figures/RL/rl_chp7_fig4.png)


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

![n-step prediction](../../figures/RL/rl_chp7_fig5.png)

#### $n$-step $Q(\sigma)$

![n-step Q(\sigma)](../../figures/RL/rl_chp7_fig6.png)


## Chapter 8 Planning and Learning with Tabular Methods

## Chapter 9 On-policy Prediction with Approximation

> Consequently, when a single state is updated, the change generalizes from that state to affect the values of many other states.

> All of the prediction methods covered in this book have been described as updates to an estimated value function that shift its value at particular states toward a “backed-up value,” or update target, for that state.

$$
S_t \rightarrow G_t
$$

$$
S_t \rightarrow R_{t+1} + \gamma V_\pi(S_{t+1}; w)
$$

$$
S_t \rightarrow G_{t:t+n}
$$

$$
S_t \rightarrow G_t^\lambda
$$

$$
s \rightarrow E_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}; w) | S_t=s]
$$

> The update $s\rightarrow u$ means that the estimated value for state s should be more like the update target u.

> Up to now, the actual update has been trivial: the table entry for s’s estimated value has simply been shifted a fraction of the way toward u, and the estimated values of all other states were left unchanged.

> Moreover, the learned values at each state were decoupled—an update at one state affected no other. But with genuine approximation, an update at one state affects many others, and **it is not possible to get the values of all states exactly correct.**

> By assumption we have far more states than weights, **so making one state’s estimate more accurate invariably means making others’ less accurate.**

> Mean Squared Value Error: often $\mu(s)$ is chosen to be the fraction of time spent in S.Under on-policy training this is called on-policy distribution. 

$$
\overline{VE}(w) = \sum_s \mu(s)(V_\pi(s) - \hat V(s; w))^2
$$

> Stochastic gradient-descent (SGD) methods do this by adjusting the weight vector after each example by a small amount in the direction that would most reduce the error on that example:

$$
\begin{array}{ll}
w_{t+1} &= w_t + \frac{1}{2} \alpha (V_\pi(s) - \hat V(s; w)) \\
 &= w_t + \alpha (V_\pi(s) - \hat V(s; w)) \nabla_w \hat V(s; w)
\end{array}
$$

> Remember that we do not seek or expect to find a value function that has zero error for all states, but only an approximation that balances the errors in different states.

> We turn now to the case in which the target output, here denoted $U_t \in R$, of the $t$-th training example, $S_t \rightarrow U_t$, is not the true value, $V_\pi(S_t)$, but some, possibly random, approximation to it.

$$
w_{t+1} = w_{t} + \alpha(U_t - V_\pi(S_t;w)) \nabla_w V_\pi(S_t; w)
$$

![Gradient MC Algorithm](../../figures/RL/rl_chp9_fig1.png)

* They take into account the effect of changing the weight vector $w_t$ on the estimate, but ignore its effect on the target. They include only a part of the gradient and, accordingly, we call them semi-gradient methods.

$$
w_{t+1} = w_{t} + \alpha (R_{t+1} + \gamma V_\pi(S_{t+1}; w) - V_\pi(S_{t}; w)) \nabla_w V_\pi(S_t; w)
$$

![Semi-gradient TD(0)](../../figures/RL/rl_chp9_fig2.png)

## Chapter 10 On-policy Control with Approximation

### Episodic Semi-gradient SARSA

$$
\delta = R_{t+1} + Q_\pi(S_{t+1}, A_{t+1}; w) - Q_\pi(S_t, A_t; w)
$$

$$
w_t = w_t + \alpha\delta \nabla_w Q_\pi(S_t, A_t; w)
$$

![Episodic semi-gradient SARSA](../../figures/RL/rl_chp10_fig1.png)

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

![Episodic semi-gradient n-step SARSA](../../figures/RL/rl_chp10_fig2.png)

### Differential Semi-gradient SARSA (using average reward for continuing problem)

> The average reward setting applies to continuing problems, problems for which the interaction between agent and environment goes on and on forever without termination or start states.

**differential return**

$$
G_t = R_{t+1} - \hat R + R_{t+2} - \hat R + R_{t+3} - \hat R + \dots
$$

**differential state-value function**

$$
V_\pi(s) = \sum_a \pi(a|s) \sum_{r, s'}p(r, s'|s, a)(r - \overline r + \gamma V_pi(s'))
$$

**differential action-value function**

$$
Q_\pi(s, a) = \sum_{r, s'} p(r, a'|s, a)(r - \overline r +\gamma V_\pi(s')))
$$

$$
Q_\pi(s, a) = \sum_{r, s'} p(r, a'|s, a)(r - \overline r +\gamma \sum_{a'}\pi(a'|s')Q(a', s')))
$$

TD errors:

$$
G_{t:t+1} = R_{t+1} - \overline R + Q_\pi(S_{t+1}, A_{t+1}; w)
$$

$$
\delta = G_{t:t+1} - Q(S_t, A_t; w)
$$

$$
w_t = w_t + \alpha \delta \nabla_wQ(S_t, A_t; w)
$$

![Differential semi-gradient SARSA](../../figures/RL/rl_chp10_fig3.png)

### Differential Semi-gradient n-step SARSA (using average reward for continuing problem)

$$
G_{t:t+n} = R_{t+1} - \overline R + R_{t+2} - \overline R + R_{t+3} - \overline R + \dots + Q_\pi(S_{t+n}; A_{t+n}; w)
$$

$$
\delta = G_{t:t+n} - Q_\pi(S_t, A_t; w)
$$

$$
w_t = w_t + \alpha \delta \nabla_w Q(S_t, A_t; w)
$$

![Differential semi-gradient n-step SARSA](../../figures/RL/rl_chp10_fig4.png)


## Chpater 12 Eligibility Traces

### $\lambda$-return

This section introduction a new kind of update target $G_t^\lambda$.

#### Prediction Problem

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n v_\pi(S_{t+1;w})
$$

> If $\lambda = 0$, then the overall update reduces to its first component, the one-step TD update, whereas if $\lambda=1$, then the overall update reduces to its last component, the Monte Carlo update.

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

![forward view](../../figures/RL/rl_chp13_fig1.png)


### TD($\lambda$)

> The weight vector is a long-term memory, accumulating over the lifetime of the system, the eligibility trace is a short-term memory, typically lasting less time than the length of an episode.

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

![Semi-gradient TD(\lambda)](../../figures/RL/rl_chp13_fig2.png)

> At each moment we look at the current TD error and assign it backward to each prior state according to how much that state contributed to the current eligibility trace at that time.

![backward view](../../figures/RL/rl_chp13_fig3.png)

> If $\lambda=0$, then by $
z_{t} = \gamma \lambda z_{t-1} + \nabla_w V_\pi (S_t; w) 
$ the trace at $t$ is exactly the value gradient corresponding to $S_t$.Thus the TD($\lambda$) update (12.7) reduces to the one-step semi-gradient TD update treated in Chapter 9 (and, in the tabular case, to the simple TD rule (6.2)). This is why that algorithm was called TD(0).

> We say that the earlier states are given less **credit** for the TD error.

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