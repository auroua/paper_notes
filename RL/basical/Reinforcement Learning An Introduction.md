# Reinforcement Learning: An Introduction

## Chpater 1 Introduction

> When an infant plays, waves its arms, or looks about, it has no explicit teacher, but it does have a direct sensorimotor connection to its environment. **Exercising this connection produces a wealth of information about cause and effect, about the consequences of actions, and about what to do in order to achieve goals.**

> **Learning from interaction is a foundational idea underlying nearly all theories of learning and intelligence.**

> Rather than directly theorizing about how people or animals learn, we primarily explore idealized learning situations and evaluate the effectiveness of various learning methods.

> The approach we explore, called reinforcement learning, is much more focused on **goal-directed learning from interaction** than are other approaches to machine learning.

### Reinforcement Learning

> Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal. **The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them.**

> These two characteristics: **trial-and-error search** and **delayed reward**—are the two most important distinguishing features of reinforcement learning.

> But the basic idea is **simply to capture the most important aspects of the real problem facing a learning agent interacting over time with its environment to achieve a goal.**  **A learning agent must be able to sense the state of its environment to some extent and must be able to take actions that affect the state. The agent also must have a goal or goals relating to the state of the environment.**

> **Reinforcement learning is trying to maximize a reward signal instead of trying to find hidden structure**. Uncovering structure in an agent’s experience can certainly be useful in reinforcement learning, but by itself does not address the reinforcement learning problem of maximizing a reward signal.

> On a stochastic task, each action must be tried many times to gain a reliable estimate of its expected reward.

### Examples

> All involve **interaction** between an active decision-making agent and its environment, within which the agent seeks to achieve a **goal** despite **uncertainty** about its environment. 

> Correct choice requeires taking into account indirect, delayed consequences of actions, and thus may require foresight or planning.

> The effects of actions cannot be fully predicted; thus the agent must monitor its environment frequently and react appropriately.

> All these examples involve goals that are explicit in the sense that the agent can judge progress toward its goal based on what it can sense directly.

### Elements of Reinforcement Learning

> A **policy**, **a reward signal**, **a value function**, and, optionally, **a model of the environment**.

> A **policy** defines the learning agent’s way of behaving at a given time. Roughly speaking, **a policy is a mapping from perceived states of the environment to actions to be taken when in those states**.

> A **reward signal** defines the goal of a reinforcenment learning problem. On each time step, the environment sends to the reinforcement learning agent a single number called the reward. **The agent’s sole objective is to maximize the total reward it receives over the long run**. The reward signal thus defines what are the good and bad events for the agent.

> The reward signal is the primary basis for altering the policy; if an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future. In general, reward signals may be stochastic functions of the state of the environment and the actions taken.

> Whereas the reward signal indicates what is good in an immediate sense, a value function specifies what is good in the long run. The value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. **Whereas rewards determine the immediate, intrinsic desirability of environmental states, values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states.**

### Limitations and Scope

> **We can think of the state as a signal conveying to the agent some sense of “how the environment is” at a particular time**.

## Chpater 2 Multi-armed Bandits

> Purely evaluative feedback indicates how good the action taken was, but not whether it was the best or the worst action possible. Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken.

> Greedy action selection always exploits current knowledge to maximize immediate reward; it spends no time at all sampling apparently inferior actions to see if they might really be better.


### Incremental Implementation

$$
Q_n = \frac{R_1 + R_2 + R_3 + \cdots + R_{n-1}}{n-1}
$$

$$
Q_{n+1} = \frac{1}{n} \sum_{i=1}^{n} R_i  \\
        = \frac{1}{n} (\sum_{i=1}^{n} R_i + R_n) \\
        = \frac{1}{n} ((n-1)Q_n + R_n) \\ 
        = \frac{1}{n}(nQ_n - Q_n + R_n ) \\
        = Q_n + \frac{1}{n}(R_n - Q_n)
$$

$$
NewEstimate \leftarrow OldEstimat + StepSize [Target - OldEstimate].
$$

> The target is presumed to **indicate a desirable direction in which to move, though it may be noisy**.

![bandit algorithm](../../figures/RL/rl_chp2_fig1.png)

$$
\sum_{n=1}^{\infty} \alpha_n(a) = \infty    \quad   and   \quad  \sum_{n=1}^{\infty} \alpha_n^2(n) < \infty
$$

> **The first condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations. The second condition guarantees that eventually the steps become small enough to assure convergence.**

### Upper-Confidence-Bound Action Selection

$$
A_t = \argmax_a [Q_t(a) + c \sqrt{\frac{lnt}{N_t(a)}}]
$$


## Chapter 3 Finite Markov Decision Processes

> MDPs are meant to be a straightforward **framing of the problem of learning from interaction to achieve a goal**. The learner and decision maker is called the **agent**.The thing it interacts with, comprising everything outside the agent, is called the **environment**.

![agent-environment interaction](../../figures/RL/rl_chp3_fig1.png)

> At each time step $t$, the agent receives some representation of the environment’s **state**, $S_t \in S$, and on that basis selects an **action**, $A_t \in A(S)$. One time step later, in part as a consequence of its action, the agent receives a numerical **reward** , $R_{t+1} \in R $, and finds itself in a new state, $S_{t+1}$.

> The MDP and agent together thereby give rise to a sequence or **trajectory** that begins like this:

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \cdots
$$

> In a **finite MDP**, the sets of states, actions, and rewards ($S$, $A$, and $R$) all have a finite number of elements. $p$ defines the dynamics of the MDP.

$$
p(s', r|s, a) = Pr\{S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a \}
$$

> In a **Markov** decision process, the probabilities given by $p$ completely characterize the environment’s dynamics.

> **State-transition probabilities**

$$
p(s'|s, a) = Pr\{S_t=s'|S_{t-1}=s, A_{t-1}=a\} = \sum_{r \in R} p(s', r|s, a)
$$

> The expected rewards for state-action pairs $r: S \times A \rightarrow R$

$$
r(s, a) = E[R_t|S_{t-1}=s, A_{t-1}=a] = \sum_{r\in R} r \sum_{s'\in S} p(s', r|s, a)
$$

> Actions can be any decisions we want to learn how to make, and the states can be anything we can know that might be useful in making them.

> The agent–environment boundary represents the limit of the agent’s **absolute control**, not of its knowledge.

> **One signal to represent the choices made by the agent (the actions), one signal to represent the basis on which the choices are made (the states), and one signal to define the agent’s goal (the rewards)**.

### Goals and Rewards

> **This means maximizing not immediate reward, but cumulative reward in the long run**. We can clearly state this informal idea as the **reward hypothesis**:

> **That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)**.

> In particular, **the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do**.

> If achieving these sorts of subgoals were rewarded, then the agent might find a way to achieve them without achieving the real goal.

> **The reward signal is your way of communicating to the robot what you want it to achieve, not how you want it achieved**.

### Returns and Episodes

> Expected return

$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T
$$

> Each episode ends in a special state called the **terminal state**. Tasks with episodes of this kind are called **episodic tasks**.

> this would be the natural way to formulate an on-going process-control task, or an application to a robot with a long life span. We call these **continuing tasks**

> The agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized. In particular, it chooses $A_t$ to maximize the **expected discounted return**: $\gamma$ is called **discount rate**.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
$$

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \\
    = R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \cdots) \\ 
    = R_{t+1} + \gamma G_{t+1}
$$

### Policy and Value Functions

> The value function of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when starting in $s$ and following $\pi$ thereafter. We call the function $v_\pi$ the state-value function for policy $\pi$.

$$
v_\pi(s) = E_\pi[G_t|S_t = s] = E_\pi[\sum_{k=0}^\infty \gamma^k R_{t+k+1} | S_t=s]
$$

> Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$, denoted $q_\pi(s, a)$, as the expected return starting from $s$, taking the action $a$, and thereafter following policy $\pi$. We call $q_\pi$ the **action-value function for policy** $\pi$.

$$
q_\pi(s, a) = E_\pi[G_t|S_t=s, A_t=a] = E_\pi[\sum_{k=0}^\infty \gamma^k R_{t+k+1} | S_t=s, A_t=a]
$$

> If an agent follows policy $\pi$ and maintains an average, for each state encountered, of the actual returns that have followed that state, then the average will converge to the state’s value, $v_\pi(s)$, as the number of times that state is encountered approaches infinity. If separate averages are kept for each action taken in each state, then these averages will similarly converge to the action values, $q_\pi(s, a)$. We call estimation methods of this kind Monte Carlo methods because they involve averaging over many random samples of actual returns.

$$
v_\pi(s) = E_\pi[G_t|S_t=s] \\ 
         = E_\pi[R_{t+1} + \gamma G_{t+1} | S_t=s] \\ 
         = \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r|s, a)[r + \gamma E_\pi[G_{t+1}|S_{t+1}=s']] \\
         = \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r|s, a)[r+\gamma v_\pi(s')], \quad for \quad all \quad s \in S
$$

> It is really a sum over all values of the three variables, $a, s'$ and $r$. For each triple, we compute its probability, $\pi(a|s)p(s_0,r|s, a)$, weight the quantity in brackets by that probability, then sum over all possibilities to get an expected value. The above equation is called **Bellman equation for** $v_\pi$. It expresses a relationship between the value of a state and the values of its successor states. It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way.

$$
q_\pi(s, a) = \sum_{s', r} p(s', r|s, a)(r+\gamma v_\pi(s')) \\ 
            = \sum_{s', r} p(s', r|s, a)(r+\gamma \sum_{a'} \pi(a'|s')q_\pi(s', a'))
$$

$$
v_\pi(s) = \sum_a \pi(a|s) q_\pi(s, a)
$$

### Optimal Policies and Optimal Value Functions

> A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ if its expected return is greater than or equal to that of $\pi'$ for all states. In other words, $\pi > \pi'$ if and only if $v_\pi(s) > v_{\pi'}(s)$ for all $s \in S$. This is an **optimal policy**. They share the same state-value function, called the **optimal state-value function**, denoted $v_\star$, and defined as 

$$
v_\star(s) = \max_\pi v_\pi(s)
$$

$$
q_\star(s) = \max_\pi q_\pi(s, a)
$$

$$
v_\star(s) = \max_{a\in A(s)} q_{\pi_\star}(s, a)  \\
          = \max_a \sum_{s', r} p(s', r| s, a)(r + \gamma v_\star(s'))
$$

$$
q_\star(s, a) = \sum_{(s', r)} p(s', r|s, a)(r + \gamma v_\star(s'))  \\ 
            = \sum_{(s', r)} p(s', r | s, a)(r + \gamma \max_{a'} q_\star(s', a'))
$$

> This solution relies on at least three assumptions that are rarely true in practice: (1) we accurately know the dynamics of the environment; (2) we have enough computational resources to complete the computation of the solution; and (3) the Markov property.

> **Many reinforcement learning methods can be clearly understood as approximately solving the Bellman optimality equation, using actual experienced transitions in place of knowledge of the expected transitions**.

### Optimality and Approximation

> In tasks with small, finite state sets, it is possible to form these approximations using arrays or tables with one entry for each state (or state–action pair). This we call the **tabular case**, and the corresponding methods we call **tabular methods**.

> **The online nature of reinforcement learning makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states. This is one key property that distinguishes reinforcement learning from other approaches to approximately solving MDPs**.

### Summary of Chapter

> Reinforcement learning is about learning from interaction how to behave in order to achieve a goal. The reinforcement learning agent and its environment interact over a sequence of discrete time steps.

> **The actions are the choices made by the agent; the states are the basis for making the choices; and the rewards are the basis for evaluating the choices. Everything inside the agent is completely known and controllable by the agent; everything outside is incompletely controllable but may or may not be completely known. A policy is a stochastic rule by which the agent selects actions as a function of states. The agent’s objective is to maximize the amount of reward it receives over time**.

> A finite Markov decision process (MDP) is an MDP with finite state, action, and (as we formulate it here) reward sets.

> The return is the function of future rewards that the agent seeks to maximize (in expected value).

> A policy’s value functions assign to each state, or state–action pair, **the expected return from that state, or state–action pair, given that the agent uses the policy**. The optimal value functions assign to each state, or state–action pair, the largest expected return achievable by any policy. A policy whose value functions are optimal is an optimal policy.

> In problems of **complete knowledge**, the agent has a complete and accurate model of the environment’s dynamics. In problems of incomplete knowledge, a complete and perfect model of the environment is not available. In most cases of practical interest there are far more states than could possibly be entries in a table, and approximations must be made. In reinforcement learning we are very much concerned with cases in which optimal solutions cannot be found but must be approximated in some way.

## Chapter 4 Dynamic Programming

> The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies **given a perfect model of the environment** as a Markov decision process (MDP).

> In fact, all of these methods can be viewed as attempts to **achieve much the same effect as DP**, only with less computation and without assuming a perfect model of the environment.

> we assume that its state, action, and reward sets, $S, A$, and $R$, are finite, and that its dynamics are given by a set of probabilities $p(s', r | s, a)$, for all $s \in S$, $a \in A(s)$, $r \in R$, and $s' \in S^+$ ($S^+$ is $S$ plus a terminal state if the problem is episodic).

> As discussed there, we can easily obtain optimal policies once we have found the optimal value functions, $v_\star$ or $q_\star$, which satisfy the Bellman optimality equations:

$$
v_\star(s) = \max_a E[R_{t+1} + \gamma v_\star(S_{t+1}) | S_{t+1}, A_t=a] \\
  = \max_a \sum_{s', r} p(s', r | s, a)(r+\gamma v_\star(s'))
$$

$$
q_\star(s, a) = E[R_{t+1} + \gamma \max_{a'}q_\star(S_{t+1}, a') | S_t=s, A_t=a] \\
     = \sum_{s', r} p(s', r | s, a)(r+\max_{a'}q_\star(s', a'))
$$

### Policy Evaluation (Prediction)

> First we consider how to compute the state-value function $v_\pi$ for an arbitrary policy $\pi$. This is called policy evaluation in the DP literature. We also refer to it as the prediction problem.

$$
v_\pi(s) = E_\pi[G_t | S_t=s] \\
    = E_\pi[R_{t+1}+\gamma G_{t+1} | S_t = s] \\
    = E_\pi[R_{t+1} + \gamma v_\pi(s_{t+1}) | S_t = s] \\
    = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a) [r + \gamma v_\pi(s')] 
$$

> The initial approximation, $v_0$, is chosen arbitrarily (except that the terminal state, if any, must be given value 0), and each successive approximation is obtained by using the Bellman equation for $v_\pi$

$$
v_{k+1}(s) = E_\pi[R_{t+1} + \gamma v_k(S_{t+1} | S_t)] \\ 
     = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)(r+\gamma v_k(s'))
$$

> Indeed, the sequence ${v_k}$ can be shown in general to converge to $v_\pi$ as $k \rightarrow \infty $ under the same conditions that guarantee the existence of $v_\pi$. This algorithm is called **iterative policy evaluation**.

> All the updates done in DP algorithms are called **expected updates** because they are based on an expectation over all possible next states rather than on a sample next state.

![iterative policy evaluation](../../figures/RL/rl_chp4_fig1.png)

### Policy Improvement

> The key criterion is whether this is greater than or less than $v_\pi(s)$. If it is greater—that is, if it is better to select a once in $s$ and thereafter follow $\pi$ than it would be to follow $\pi$ all the time—then one would expect it to be better still to select a every time s is encountered, and that the new policy would in fact be a better one overall.

> That this is true is a special case of a general result called the **policy improvement theorem**.

> Let $\pi$ and $\pi'$ be any pair of deterministic policies such that, for all $s \in S$,

$$
q_\pi(s, \pi'(s)) \geq v_\pi(s)
$$

> Then the policy $\pi'$ must be as good as, or better than, $\pi$. That is, it must obtain greater or equal expected return from all states $s \in S$:

$$
v_{\pi'}(s) \geq v_\pi(s)
$$

> The process of making a new policy that improves on an original policy, by making it greedy with respect to the value function of the original policy, is called **policy improvement**.

### Policy Iteration

$$
\pi_0 \rightarrow v_{\pi_0} \rightarrow \pi_1 \rightarrow v_{\pi_1} \rightarrow \pi_2 \rightarrow v_{\pi_2} \rightarrow \cdots \pi_\star \rightarrow v_{\pi_\star}
$$

![policy iteration](../../figures/RL/rl_chp4_fig2.png)

### value iteration

$$
v_{k+1}(s) = \max_a E[R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s, A_t = a] \\ 
    = \max_a p(s', r|s, a) (r + \gamma v_k(s'))
$$

> Note that value iteration is obtained simply by turning the Bellman optimality equation into an update rule.

![value iteration](../../figures/RL/rl_chp4_fig3.png)


### Asynchronous Dynamic Programming

> **Asynchronous DP algorithms** are in-place iterative DP algorithms that are not organized in terms of systematic sweeps of the state set.

> **These algorithms update the values of states in any order whatsoever, using whatever values of other states happen to be available. The values of some states may be updated several times before the values of others are updated once.**

> For example, one version of asynchronous value iteration updates the value, in place, of only one state, $s_k$, on each step, $k$, using the value iteration update.

> It just means that an algorithm does not need to **get locked into any hopelessly long sweep before it can make progress improving a policy**. We can try to take advantage of this flexibility by **selecting the states to which we apply updates so as to improve the algorithm’s rate of progress**. We can try to **order the updates to let value information propagate from state to state in an efficient way**. Some states may not need their values updated as often as others. We might even **try to skip updating some states entirely if they are not relevant to optimal behavior**.

> We can apply updates to states as the agent visits them. **This makes it possible to focus the DP algorithm’s updates onto parts of the state set that are most relevant to the agent**. Ref: Trajectory Sampling

### Generalized Policy Iteration

> Policy iteration consists of two simultaneous, interacting processes, **one making the value function consistent with the current policy (policy evaluation)**, and the other **making the policy greedy with respect to the current value function (policy improvement)**.

> We use the term generalized policy iteration (GPI) to refer to the general idea of letting policy-evaluation and policy-improvement processes interact, independent of the granularity and other details of the two processes. Almost all reinforcement learning methods are well described as GPI. That is, all have identifiable policies and value functions, with the **policy always being improved with respect to the value function** and **the value function always being driven toward the value function for the policy**.

![GPI](../../figures/RL/rl_chp4_fig4.png)

> The value function stabilizes only when it is consistent with the current policy, and the policy stabilizes only when it is greedy with respect to the current value function.

> The evaluation and improvement processes in GPI can be viewed as both **competing and cooperating**. They compete in the sense that they pull in opposing directions. **Making the policy greedy with respect to the value function typically makes the value function incorrect for the changed policy**, and **making the value function consistent with the policy typically causes that policy no longer to be greedy**.

> Each process drives the value function or policy toward one of the lines representing a solution to one of the two goals. The goals interact because the two lines are not orthogonal. Driving directly toward one goal causes some movement away from the other goal.

![GPI](../../figures/RL/rl_chp4_fig5.png)

### Summary of Chapter

> GPI is the general idea of two interacting processes revolving around an approximate policy and an approximate value function. One process takes the policy as given and performs some form of policy evaluation, changing the value function to be more like the true value function for the policy. The other process takes the value function as given and performs some form of policy improvement, changing the policy to make it better, assuming that the value function is its value function.

> All of them update estimates of the values of states based on estimates of the values of successor states. That is, they update estimates on the basis of other estimates. We call this general idea **bootstrapping**.

## Chapter 5 Monte Carlo Methods


> Monte Carlo methods require only **experience**—sample sequences of states, actions, and rewards from **actual or simulated interaction** with an environment.

### Monte Carlo Prediction

> An obvious way to estimate it from experience, then, is simply to average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. This idea underlies all Monte Carlo methods.

> The first-visit MC method estimates $v_\pi(s)$ as the average of the returns following first visits to $s$, whereas the every-visit MC method averages the returns following all visits to $s$.

![first visit mc prediction](../../figures/RL/rl_chp5_fig1.png)

> In particular, note that the computational expense of **estimating the value of a single state is independent of the number of states.** This can make Monte Carlo methods particularly attractive when one requires the value of only one or a subset of states. **One can generate many sample episodes starting from the states of interest, averaging returns from only these states, ignoring all others.** This is a third advantage Monte Carlo methods can have over DP methods.

### Monte Carlo Estimation of Action Values

> With a model, state values alone are sufficient to determine a policy; one simply looks ahead one step and chooses whichever action leads to the best combination of reward and next state.

> To compare alternatives we need to estimate the value of all the actions from each state, not just the one we currently favor. This is the general problem of **maintaining exploration.** One way to do this is by specifying that **the episodes start in a state–action pair, and that every pair has a nonzero probability of being selected as the start.** We call this the assumption of **exploring starts.**

> The most common alternative approach to assuring that all state–action pairs are encountered is to consider only policies that are stochastic with a nonzero probability of selecting all actions in each state.


### Monte Carlo Control

> The observed returns are used for policy evaluation, and then the policy is improved at all the states visited in the episode.

![mces](../../figures/RL/rl_chp5_fig2.png)


### Monte Carlo Control without Exploring Starts

> There are two approaches to ensuring this, resulting in what we call **on-policy methods** and **offpolicy methods**. **On-policy methods attempt to evaluate or improve the policy that is used to make decisions, whereas offpolicy methods evaluate or improve a policy different from that used to generate the data.**

> In on-policy control methods the policy is generally **soft**, meaning that $\pi(a|s)>0$ for all $s\in S$ and all $a \in A(s)$, but gradually shifted closer and closer to a deterministic optimal policy.

> The $\epsilon$-greedy policies are examples of $\epsilon$-soft policies, defined as policies for which $\pi(a|s) \geq \frac{\epsilon}{|A(s)|}$ for all states and actions, for some $\epsilon > 0$.

![on-policy first-visit mc control](../../figures/RL/rl_chp5_fig3.png)

### Off-policy Prediction via Importance Sampling

> A more straightforward approach is to use two policies, one that is learned about and that becomes the optimal policy, and one that is more exploratory and is used to generate behavior. The policy being learned about is called the **target policy**, and the policy used to generate behavior is called the **behavior policy.** In this case we say that learning is from data “off” the target policy, and the overall process is termed **off-policy learning**.

> In order to use episodes from $b$ to estimate values for $\pi$, we require that every action taken under $\pi$ is also taken, at least occasionally, under $b$. That is, we require that $\pi(a|s)>0$ implies $b(a|s)>0$. This is called the assumption of **coverage**. It follows from coverage that $b$ must be stochastic in states where it is not identical to $\pi$.
 
> Given a starting state $S_t$, the probability of the subsequent state–action trajectory, $A_t, S_{t+1}, A_{t+1}, \cdots, S_T$ , occurring under any policy $\pi$ is

$$
Pr\{A_t, S_{t+1}, A_{t+1}, \cdots, S_T | S_t, A_{t:T-1} \sim \pi \}  \\
     = \pi(A_t|S_t)p(S_{t+1}|S_t, A_t)\pi(A_{t+1}|S_{t+1})\cdots p(S_T|S_{T-1}, A_{T-1}) \\
     = \prod_{k=t}^{T-1} \pi(A_k | S_k) p(S_{k+1}|S_k, A_k)
$$

$$
\begin{array}{lll}
\rho_{t:T-1} &= \frac{\prod_{k=t}^{T-1}\pi(A_k | S_k) p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1}b(A_k | S_k) p(S_{k+1}|S_k, A_k)} \\
&= \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
\end{array}
$$

$$
E[\rho_{t:T-1}G_t|S_t=s] = v_\pi(s)
$$

**ordinary importance sampling, no biases and high variances.**

$$
V(s) = \frac{\sum_{t\in J(s)}\rho_{t:T(t)-1}G_t}{|J(s)|}
$$

**weighted importance sampling, biases and low variances.**

$$
V(s) = \frac{\sum_{t\in J(s)}\rho_{t:T(t)-1}G_t}{\sum_{t \in J(s)}\rho_{t:T(t)-1}}
$$

> The variance of ordinary importance sampling is in general unbounded because the variance of the ratios can be unbounded, whereas in the weighted estimator the largest weight on any single return is one.

> In practice, every-visit methods are often preferred because they remove the need to keep track of which states have been visited and because they are much easier to extend to approximations.

### Summary of Chapter

> First, **they can be used to learn optimal behavior directly from interaction with the environment, with no model of the environment’s dynamics.** Second, **they can be used with simulation or sample models.** For surprisingly many applications it is easy to simulate sample episodes even though it is difficult to construct the kind of explicit model of transition probabilities required by DP methods. Third, **it is easy and efficient to focus Monte Carlo methods on a small subset of the states.** A region of special interest can be accurately evaluated without going to the expense of accurately evaluating the rest of the state set. A fourth advantage of Monte Carlo methods, which we discuss later in the book, is that **they may be less harmed by violations of the Markov property.** This is because they do not update their value estimates on the basis of the value estimates of successor states. In other words, it is because they do not bootstrap.


## Chapter 6 Temporal-Difference Learning

> TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

> Monte Carlo methods wait until the return following the visit is known, then use that return as a target for $V(S_t)$.

$$
V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]
$$
> where $G_t$ is the actual return following time $t$, and $\alpha$ is a constant step-size parameter. This method is called *constant-$\alpha$ MC*.

> TD methods need to wait only until the next time step. At time $t+1$ they immediately form a target and make a useful update using the observed reward $R_{t+1}$ and the estimate $V(S_{t+1})$.

$$
V(S_{t}) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

![Tabular TD(0)](../../figures/RL/rl_chp6_fig1.png)

> **The target for the Monte Carlo update is $G_t$, whereas the target for the TD update is $R_{t+1} + \gamma V(S_{t+1})$.** This TD method is called TD(0), or one-step TD, because it is a special case of the TD($\lambda$) and n-step TD methods developed in Chapter 12 and Chapter 7.

$$
\begin{array}{lll}
v_\pi(s) &= E_\pi[G_t|S_t=s]   \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots  (6.3)\\
 &= E_\pi[R_{t+1} + \gamma G_{t+1} | S_t=s] \\ 
 &= E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t=s] \cdots \cdots \cdots   (6.4)
\end{array}
$$

> Roughly speaking, **Monte Carlo methods use an estimate of (6.3) as a target, whereas DP methods use an estimate of (6.4) as a target. The Monte Carlo target is an estimate because the expected value in (6.3) is not known; a sample return is used in place of the real expected return. The DP target is an estimate not because of the expected values, which are assumed to be completely provided by a model of the environment, but because $V_\pi(S_{t+1})$ is not known and the current estimate, $V_\pi(S_{t+1})$, is used instead. The TD target is an estimate for both reasons: it samples the expected values in (6.4) and it uses the current estimate $V$ instead of the true $v_\pi$. Thus, TD methods combine the sampling of Monte Carlo with the bootstrapping of DP.**

> *Sample updates* differ from the *expected updates* of DP methods in that they are based on a single sample successor rather than on a complete distribution of all possible successors.

> TD methods have an advantage over DP methods in that they do not require a model of the environment, of its reward and next-state probability distributions.

> With Monte Carlo methods one must wait until the end of an episode, because only then is the return known, whereas with TD methods one need wait only one time step.

> The other reasonable answer is simply to observe that we have seen $A$ once and the return that followed it was 0; we therefore estimate $V(A)$ as 0. This is the answer that batch Monte Carlo methods give. Notice that it is also the answer that gives minimum squared error on the training data. We expect that the first answer will produce lower error on future data, even though the Monte Carlo answer is better on the existing data.

> **Batch Monte Carlo methods always find the estimates that minimize mean-squared error on the training set, whereas batch TD(0) always finds the estimates that would be exactly correct for the maximum-likelihood model of the Markov process.**

### SARSA: On-policy TD Control

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

![SARSA](../../figures/RL/rl_chp6_fig2.png)


> This rule uses every element of the quintuple of events, ($S_t$,$A_t$,$R_{t+1}$,$S_{t+1}$,$A_{t+1}$), that make up a transition from one state–action pair to the next.

### Q-Learning: Off-policy TD Control

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \max_a Q(S_{t+1}, a) - Q(S_t, A_t))
$$

![Q-learning](../../figures/RL/rl_chp6_fig3.png)

### Expected Sarsa

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \sum_a \pi(a|S_t) Q(S_t, a) - Q(S_t, A_t))
$$

### Maximization Bias and Double Learning

> One way to view the problem is that it is due to using the same samples (plays) both to determine the maximizing action and to estimate its value.

$$
Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t)+ \alpha(R_{t+1} + \gamma Q_2(S_{t+1}, \argmax_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t) )
$$

![Double Q-learning](../../figures/RL/rl_chp6_fig4.png)

### The backup diagrams of the above algorithms

![Backup diagrams](../../figures/RL/rl_chp6_fig5.png)

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

> These are respectively called model-based and model-free reinforcement learning methods. Model-based methods rely on planning as their primary component, while model-free methods primarily rely on learning.

> By a *model* of the environment we mean anything that an agent can use to predict how the environment will respond to its actions.

> Some models produce a description of all possibilities and their probabilities; these we call *distribution models*. Other models produce just one of the possibilities, sampled according to the probabilities; these we call *sample models*.

> Given a starting state and action, a sample model produces a possible transition, and a distribution model generates all possible transitions weighted by their probabilities of occurring.
Given a starting state and a policy, a sample model could produce an entire episode, and a distribution model could generate all possible episodes and their probabilities.
In either case, we say the model is used to simulate the environment and produce simulated experience. 

> The word planning is used in several different ways in different fields. We use the term to refer to any computational process that takes a model as input and produces or improves a policy for interacting with the modeled environment.

> *State-space planning*, which includes the approach we take in this book, is viewed primarily as a search through the state space for an optimal policy or an optimal path to a goal.

> It takes the rest of the chapter to develop this view, but there are two basic ideas: (1) all state-space planning methods involve computing value functions as a key intermediate step toward improving the policy, and (2) they compute value functions by updates or backup operations applied to simulated experience.

> **The heart of both learning and planning methods is the estimation of value functions by backing-up update operations. The difference is that whereas planning uses simulated experience generated by a model, learning methods use real experience generated by the environment.**

![Random-sample one-step tabular Q-planning](../../figures/RL/rl_chp8_fig1.png)

### Dyna: Integrated Planning, Acting, and Learning

> **New information gained from the interaction may change the model and thereby interact with planning.**

> Within a planning agent, there are at least two roles for real experience: **it can be used to improve the model** (to make it more accurately match the real environment) and **it can be used to directly improve the value function and policy using the kinds of reinforcement learning methods we have discussed in previous chapters.** The former we call *model-learning*, and the latter we call *direct reinforcement learning* (direct RL).

> Indirect methods often make fuller use of a limited amount of experience and thus achieve a better policy with fewer environmental interactions. On the other hand, direct methods are much simpler and are not affected by **biases in the design of the model**.

> The model is learned from real experience and gives rise to simulated experience. We use the term *search control* to refer to the process that selects the starting states and actions for the simulated experiences generated by the model. Finally, planning is achieved by applying reinforcement learning methods to the simulated experiences just as if they had really happened.

> Learning and planning are deeply integrated in the sense that they share almost all the same machinery, di↵ering only in the source of their experience.

![The general Dyna Architecture](../../figures/RL/rl_chp8_fig2.png)

> In the pseudocode algorithm for Dyna-Q in the box below, $Model(s, a)$ denotes the contents of the (predicted next state and reward) for state–action pair $(s, a)$.

![Tabular Dyna-Q](../../figures/RL/rl_chp8_fig3.png)

> In Dyna-Q, learning and planning are accomplished by exactly the same algorithm, operating on real experience for learning and on simulated experience for planning.
> As new information is gained, the model is updated to better match reality. As the model changes, the ongoing planning process will gradually compute a different way of behaving to match the new model.

### When the Model Is Wrong

> Models may be incorrect because the environment is stochastic and only a limited number of samples have been observed, or because the model was learned using function approximation that has generalized imperfectly, or simply because the environment has changed and its new behavior has not yet been observed.

> In a planning context, exploration means trying actions that improve the model, whereas exploitation means behaving in the optimal way given the current model. We want the agent to explore to find changes in the environment, but not so much that performance is greatly degraded.

> To encourage behavior that tests long-untried actions, a special “bonus reward” is given on simulated experiences involving these actions. In particular, if the modeled reward for a transition is $r$, and the transition has not been tried in $\tau$ time steps, then planning updates are done as if that transition produced a reward of $r+k\sqrt \tau$ , for some small $k$.

### Prioritized Sweeping

> Simulated transitions are started in state–action pairs selected uniformly at random from all previously experienced pairs. But a uniform selection is usually not the best; planning can be much more efficient if simulated transitions and updates are focused on particular state–action pairs.

> If simulated transitions are generated uniformly, then many wasteful updates will be made before stumbling onto one of these useful ones. In the much larger problems that are our real objective, the number of states is so large that an unfocused search would be extremely inefficient.

> Suppose now that the agent discovers a change in the environment and changes its estimated value of one state, either up or down. Typically, this will imply that the values of many other states should also be changed, but the only useful one-step updates are those of actions that lead directly into the one state whose value has been changed. This general idea might be termed *backward focusing of planning computations*.

![Prioritized sweeping](../../figures/RL/rl_chp8_fig11.png)

> We have suggested in this chapter that all kinds of state-space planning can be viewed as **sequences of value updates, varying only in the type of update, expected or sample, large or small, and in the order in which the updates are done.**

> For example, another would be to focus on states according to how easily they can be reached from the states that are visited frequently under the current policy, which might be called forward focusing.


### Expected VS Sample Updates

![Expcected and Sample updates](../../figures/RL/rl_chp8_fig12.png)

> In the absence of a distribution model, expected updates are not possible, but sample updates can be done using sample transitions from the environment or a sample model.

Sample Update

$$
Q(S_t=s, A_t=a) \leftarrow Q(S_t=s, A_t=a) + \alpha (R_{t+1} + \gamma Q(S_{t+1}=s', A_{t+1}=a') - Q(S_t=s, A_t=a)) 
$$

Expected Update

$$
Q(S_t=s, A_t=a) = \sum_{s', r} p(S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a)(r+\gamma \sum_{a'}\pi(A_{t+1}=a'|S_{t+1}=s')Q(S_{t+1}=s', A_{t+1}=a'))
$$

> **In favor of the expected update is that it is an exact computation, resulting in a new $Q(s, a)$ whose correctness is limited only by the correctness of the $Q(s', a')$ at successor states. The sample update is in addition affected by sampling error. On the other hand, the sample update is cheaper computationally because it considers only one next state, not all possible next states.**

> For a particular starting pair, $s$, $a$,let $b$ be the **branching factor** (i.e., the number of possible next states, $s'$, for which $ \hat p(s' |s, a) > 0)$. Then an expected update of this pair requires roughly $b$ times as much computation as a sample update.

> **If there is enough time to complete an expected update, then the resulting estimate is generally better than that of b sample updates because of the absence of sampling error. But if there is insuffcient time to complete an expected update, then sample updates are always preferable because they at least make some improvement in the value estimate with fewer than $b$ updates.**

> In a real problem, the values of the successor states would be estimates that are themselves updated. By causing estimates to be more accurate sooner, sample updates will have a second advantage in that **the values backed up from the successor states will be more accurate**. These results suggest that sample updates are likely to be superior to expected updates on problems with large stochastic branching factors and too many states to be solved exactly.

### Trajectory Sampling

> The classical approach, from dynamic programming, is to perform sweeps through the entire state (or state–action) space, updating each state (or state–action pair) once per sweep.

> This is problematic on large tasks because there may not be time to complete even one sweep. In many tasks the vast majority of the states are irrelevant because they are visited only under very poor policies or with very low probability.

> The second approach is to sample from the state or state–action space according to some distribution.

> More appealing is to distribute updates according to the on-policy distribution, that is, according to the distribution observed when following the current policy.

> **In either case, sample state transitions and rewards are given by the model, and sample actions are given by the current policy. One simulates explicit individual trajectories and performs updates at the state or state–action pairs encountered along the way. We call this way of generating experience and updates trajectory sampling.**

> In the short term, sampling according to the on-policy distribution helps by focusing on states that are near descendants of the start state. If there are many states and a small branching factor, this effect will be large and long-lasting. In the long run, focusing on the on-policy distribution may hurt because the commonly occurring states all already have their correct values. Sampling them is useless, whereas sampling other states may actually perform some useful work. This presumably is why the exhaustive, unfocused approach does better in the long run, at least for small problems.

> They do suggest that sampling according to the on-policy distribution can be a great advantage for large problems, **in particular for problems in which a small subset of the state–action space is visited under the on-policy distribution.**

### Planning at Decision Time

> The one we have considered so far in this chapter, typified by dynamic programming and Dyna, is to use planning to gradually improve a policy or value function **on the basis of simulated experience** obtained from a model (either a sample or a distribution model).

> These two ways of thinking about planning—using **simulated experience** to gradually improve a policy or value function, or using **simulated experience** to select an action for the current state.

> **Decision-time planning** is most useful in applications in which fast responses are not required. In chess playing programs, for example, one may be permitted seconds or minutes of computation for each move, and strong programs may plan dozens of moves ahead within this time. On the other hand, if low latency action selection is the priority, then one is generally better off doing planning in the background  (**background planning**) to compute a policy that can then be rapidly applied to each newly encountered state.

### Heuristic Search

> In heuristic search, for each state encountered, a large tree of possible continuations is considered. The approximate value function is applied to the leaf nodes and then backed up toward the current state at the root.

> The backing up stops at the state–action nodes for the current state. Once the backed-up values of these nodes are computed, the best of them is chosen as the current action, and then all backed-up values are discarded.

> To compute the greedy action given a model and a state-value function, we must look ahead from each possible action to each possible next state, take into account the rewards and estimated values, and then pick the best action. Just as in conventional heuristic search, this process computes backed-up values of the possible actions, but does not attempt to save them.

> The point of searching deeper than one step is to obtain better action selections. If one has a perfect model and an imperfect action-value function, then in fact **deeper search** will usually yield better policies. **The deeper the search, the more computation is required, usually resulting in a slower response time.**.

> **Much of the effectiveness of heuristic search is due to its search tree being tightly focused on the states and actions that might immediately follow the current state.**

> **This great focusing of memory and computational resources on the current decision is presumably the reason why heuristic search can be so effective.**

> Any state-space search can be viewed in this way as the piecing together of a large number of individual one-step updates. Thus, the performance improvement observed with deeper searches is not due to the use of multistep updates as such. Instead, **it is due to the focus and concentration of updates on states and actions immediately downstream from the current state.**

![Heuristic Search](../../figures/RL/rl_chp8_fig13.png)

### Rollout Algorithms

> Rollout algorithms are decision-time planning algorithms **based on Monte Carlo control** applied to simulated trajectories that all begin at the current environment state.

> They estimate action values for a given policy by averaging the returns of **many simulated trajectories** that **start with each possible action and then follow the given policy**. When the action-value estimates are considered to be accurate enough, **the action (or one of the actions) having the highest estimated value is executed**, after which the process is carried out anew from the resulting next state.

> The goal of a rollout algorithm is not to estimate a complete optimal action-value function, $q_\star$, or a complete action-value function, $q_\pi$, for a given policy $\pi$. Instead, they produce Monte Carlo estimates of action values only for **each current state** and for a given policy usually called the **rollout policy**. As decision-time planning algorithms, rollout algorithms make immediate use of these action-value estimates, then discard them. The aim of a rollout algorithm is to improve upon the rollout policy; not to find an optimal policy. 

> Intuition suggests that the better the rollout policy and the more accurate the value estimates, the better the policy produced by a rollout algorithm is likely be.

> The number of actions that have to be evaluated for each decision, the number of time steps in the simulated trajectories needed to obtain useful sample returns, the time it takes the rollout policy to make decisions, and the number of simulated trajectories needed to obtain good Monte Carlo action-value estimates.

### Monte Carlo Tree Search

> MCTS is a rollout algorithm as described above, but enhanced by the addition of a means for **accumulating value estimates obtained from the Monte Carlo simulations in order to successively direct simulations toward more highly-rewarding trajectories.**

> The core idea of MCTS is to successively **focus multiple simulations starting at the current state by extending the initial portions of trajectories that have received high evaluations from earlier simulations.**

> As in any tabular Monte Carlo method, the value of a state–action pair is estimated as the average of the (simulated) returns from that pair. **Monte Carlo value estimates are maintained only for the subset of state–action pairs that are most likely to be reached in a few steps, which form a tree rooted at the current state.**

> **MCTS incrementally extends the tree by adding nodes representing states that look promising based on the results of the simulated trajectories.** Outside the tree and at the leaf nodes the rollout policy is used for action selections.

![MCTS](../../figures/RL/rl_chp8_fig14.png)

### Chapter Summary

> Another important dimension is the **distribution of updates**, that is, of the **focus of search**. **Prioritized sweeping focuses backward on the predecessors of states whose values have recently changed. On-policy trajectory sampling focuses on states or state–action pairs that the agent is likely to encounter when controlling its environment.**

### Part I Summary

> All of the methods we have explored so far in this book have three key ideas in common: first, **they all seek to estimate value functions**; second, **they all operate by backing up values along actual or possible state trajectories**; and third, **they all follow the general strategy of generalized policy iteration (GPI)**, meaning that **they maintain an approximate value function and an approximate policy, and they continually try to improve each on the basis of the other.**

> We suggest that **value functions, backing up value updates, and GPI** are powerful organizing principles potentially relevant to any model of intelligence, whether artificial or natural.

> The horizontal dimension is whether they are **sample updates** (based on a sample trajectory) or **expected updates** (based on a distribution of possible trajectories). **Expected updates require a distribution model, whereas sample updates need only a sample model, or can be done from actual experience with no model at all (another dimension of variation).** The vertical dimension of Figure 8.11 corresponds to **the depth of updates**, that is, to the degree of bootstrapping.

> At three of the four corners of the space are the three primary methods for estimating values: **dynamic programming**, **TD**, and **Monte Carlo**. Along the left edge of the space are the sample-update methods, ranging from one-step TD updates to full-return Monte Carlo updates. Between these is a spectrum including methods based on n-step updates.

![MCTS](../../figures/RL/rl_chp8_fig15.png)

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

> Up to now, the actual update has been trivial: the table entry for $S$’s estimated value has simply been shifted a fraction of the way toward $U$, and the estimated values of all other states were left unchanged.

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

The target of policy gradient methods and the on-policy value approximation methods are listed below.

$$
L(w) = E_{s, a}[Q_\pi(s, a) - \hat Q(s, a; w)]^2
$$

$$
J(\theta) = E_{s_0 \in \mu(s_0)}[V_\pi(s_0)]
$$

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta)
$$

> Methods that learn approximations to both policy and value functions are often called actor–critic methods, where ‘actor’ is a reference to the learned policy, and ‘critic’ refers to the learned value function, usually a state-value function.

### Policy Gradient Theorem

$$
\nabla_\theta v_\pi(s) = \sum_{x \in S}\sum_{k=0}^\infty Pr(s \rightarrow x, k, \pi) \sum_a \nabla_\theta \pi_\theta(a|x) q_\pi(x, a)
$$

$$
\nabla J(\theta) = \nabla v_\pi(S_0) \varpropto  \sum_s \mu(s) \sum_a \nabla \pi(a|s) Q_\pi(s, a)
$$

$$
\mu(s) = \frac{\eta(s)}{\sum_{s'}\eta(s')}
$$

$$
\eta(s) = \sum_{k=0}^\infty Pr(s_0 \rightarrow s, k, \pi)
$$

> $Pr(s_0 \rightarrow s, k, \pi)$ is the probability of transitioning from state $s_0$ to state $s$ in $k$ steps under policy.

$$
\nabla J(\theta) \varpropto  \sum_s \mu(s) \sum_a \nabla \pi(a|s) Q_\pi(s, a)
$$

### REINFORCE: Monte Carlo Policy Gradient

$$
\nabla J(\theta) \varpropto  \sum_s \mu(s) \sum_a \nabla \pi(a|s) Q_\pi(s, a) \\
$$

$$
\nabla J(\theta) = E_{\mu(s)}[\sum_a Q_\pi(s, a)\nabla \pi(a|s; \theta)]
$$

$$
\theta_{t+1} = \theta_t + \alpha \sum_a  Q(S_t, a) \nabla \pi(a|S_t; \theta)
$$

> This algorithm, which has been called an **all-actions method** because its update involves all of the actions, is promising and deserving of further study, but our current interest is the classical **REINFORCE algorithm** (Willams, 1992) whose update at time $t$ involves just $A_t$, the one action actually taken at time $t$.

$$
\begin{array}{ll}
\nabla J(\theta) &= E_{s \in \mu(s)}\lbrack\sum_a \pi(a|s; \theta) Q_\pi(s, a) \frac{\nabla \pi(a|s; \theta)}{\pi(a|s; \theta)}\rbrack \\
&=E_{s \in \mu(s), a \in \pi(a|s)}[Q_\pi(s, a) \frac{\nabla \pi(a|s; \theta)}{\pi(a|s; \theta)}] \\
&= E_{s \in \mu(s), a \in \pi(a|s)}[G_t \frac{\nabla \pi(a|S_t; \theta)}{\pi(a|S_t; \theta)}]
\end{array}
$$

**REINFORCE update**

$$
\theta_{t+1} = \theta_t + \alpha G_t \frac{\nabla \pi(a|S_t; \theta)}{\pi(a|S_t; \theta)}
$$

> The vector is the direction in parameter space that most increases the probability of repeating the action $A_t$ on future visits to state $S_t$. The update increases the parameter vector in this direction proportional to the return, and inversely proportional to the action probability. The former makes sense because it causes the parameter to move most in the directions that favor actions that yield the highest return. The latter makes sense because otherwise actions that are selected frequently are at an advantage (the updates will be more often in their direction) and might win out even if they do not yield the highest return.

![REINFORCE MC Update](../../figures/RL/rl_chp13n_fig1.png)

### REINFORCE with Baseline

$$
\nabla J(\theta) \varpropto  \sum_s \mu(s) \sum_a (Q_\pi(s, a)-b(s)) \nabla \pi(a|s; \theta)
$$

$$
\sum_a b(s) \nabla \pi(a|s; \theta) = b(s) \sum_a \nabla \pi(a|s; \theta) = b(s) \nabla 1 = 0
$$

$$
\theta_{t+1} = \theta_t + \alpha(G_t - b(S_t)) \frac{\nabla \pi(a|S_t; \theta)}{\pi(a|S_t; \theta)}
$$

> In general, the baseline leaves the expected value of the update unchanged, but it can have a large e↵ect on its variance.

> In some states all actions have high values and we need a high baseline to di↵erentiate the higher valued actions from the less highly valued ones; in other states all actions will have low values and a low baseline is appropriate.

![REINFORCE with Baseline](../../figures/RL/rl_chp13n_fig2.png)

### Actor-Critic Methods

> Although the REINFORCE-with-baseline method learns both a policy and a state-value function, we do not consider it to be an actor–critic method because its state-value function is used only as a baseline, not as a critic. That is, it is not used for bootstrapping (updating the value estimate for a state from the estimated values of subsequent states), but only as a baseline for the state whose estimate is being updated.


$$
\begin{array}{ll}
\theta_{t+1} &= \theta_t + \alpha (G_{t:t+1} - \hat v(S_t; w)) \frac{\nabla \pi(a|S_t; \theta)}{\pi(a|S_t; \theta)} \\
&= \theta_t + \alpha (R_{t+1} + \gamma V_\pi(S_{t+1}; w) - \hat V_\pi(S_t; w)) \frac{\nabla \pi(a|S_t; \theta)}{\pi(a|S_t; \theta)} \\
&= \theta_t + \alpha \delta_t \frac{\nabla \pi(a|S_t; \theta)}{\pi(a|S_t; \theta)} \\
\end{array}
$$

![Acotr-Critic Episodic](../../figures/RL/rl_chp13n_fig3.png)

![Acotr-Critic with Eligibility Traces Episodic](../../figures/RL/rl_chp13n_fig4.png)

### Policy Gradient for Continuing Problems

![Acotr-Critic with Eligibility Traces Continuing](../../figures/RL/rl_chp13n_fig5.png)

### Policy Parameterization for Continuous Actions

$$
p(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

![Gaussian Distribution](../../figures/RL/rl_chp13n_fig6.png)

$$
\pi(a|s; \theta) = \frac{1}{\sigma(s; \theta) \sqrt{2\pi}} \exp(-\frac{(x-\mu(s;\theta))^2}{2\sigma(s;\theta)^2})
$$