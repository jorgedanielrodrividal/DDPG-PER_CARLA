# DDPG-PER in CARLA

This project implements Deep Deterministic Policy Gradients (DDPG) with Exploration [1] and Prioritized Experience Replay (PER) [2] in CARLA simulator [3]. The Reinforcement Learning agent is able to learn a policy for lane following faster as compared to Deep Q Learning [4] agent. Seven agent variants are implemented, six of them using no image as input and one using B/W segmented image of the road over the whole route that the vehicle must drive. A priority to waypoint input is given in this study over image input is done due to the former being a much quicker approach  This study is heavily inspired by the pioneering research of Kendall et. al. [5], who implemented a DDPG-PER trained exclusively with a single monocular image in a real electric vehicle and Pérez‑Gil et. al. [6] who implemented several DDPG agents in CARLA. This implementation is based on the [official implementation](https://github.com/RobeSafe-UAH/DDPG-CARLA) of [6]

## Setup


- Intel i7 gen 9th - 11th / Intel i9 gen 9th - 11th / AMD Ryzen 7 / AMD Ryzen 9
- +32 Gb RAM memory
- NVIDIA RTX 3070/3080/3090 / NVIDIA RTX 4090 or better
- 16 Gb or more VRAM
- CARLA 0.9.14
- Python 3.7
- Numpy
- Tensorflow==1.14.0
- Keras==2.2.4
- OpenCV==4.1.2

## Train & Evaluate

For training set `TRAIN_PLAY == 1` in `config.py` and run

```
launch_DDPG.py
```

...

## Related Work 

The closest work in the current literature can be clasified as either classical autonomous driving relying on mapping, imitation learning and Reinforcement Learning.

### 1) Classical Autonomous Driving. 

Several landmark challenges, such as the DARPA Urban Challenge and the Intelligent Vehicle Future Challenge (IVFC), have significantly advanced the field of autonomous driving by evaluating systems in complex, real-world scenarios that demand sophisticated sensing and control strategies [7]. These approaches typically rely on what is known as a modular pipeline, in which the driving system is composed of independently designed components responsible for specific functions, such as low-level perception, scene parsing, mapping, planning, and control [8]. A key advantage of this architecture lies in its modularity, which facilitates parallel development and system integration. However, this same modularity introduces a major limitation: the inability to train the entire system in an end-to-end fashion. Each module is optimized independently, often using training data that is not representative of the complete driving task, potentially leading to suboptimal overall performance. Furthermore, a notable drawback of these pipelines is their reliance on high-definition (HD) maps for both localization and planning (resources that are costly to generate and maintain). Consequently, there is growing interest in localization techniques that do not depend on HD maps [9]. For instance, recent work has employed deconvolutional neural networks to detect structural changes in street-view video, enabling more scalable and frequent updates to large-scale maps [10]. In addition, semantic segmentation remains a cornerstone of low-level perception, providing critical scene understanding capabilities [11], [12], [13]. These modular perception and mapping frameworks continue to underpin commercial efforts aimed at building robust and scalable autonomous driving systems.

### 2) End-to-End learning: Imitation learning. 

Autonomous lane following based on imitation learning long history [14]. Imitation learning aims to train a policy to mimic given expert decisions. An imitation learning framework consists of:

- a set $\mathcal{S}$ of states, typically only partially observed,  
- a set $\mathcal{A}$ of actions, which may be:
  - discrete: actions are mapped to a fixed set of numbers (e.g., `turn_left`, `turn_right`), or  
  - continuous: actions take values in a range (typically $[-1, 1]$) and are not associated with a fixed action set,  
- a policy $\pi_{\theta} : \mathcal{S} \rightarrow \mathcal{A}$, parameterized by $\theta$, that is learned to imitate the expert,  
- an expert policy $\pi_{\theta}^* : \mathcal{S} \rightarrow \mathcal{A}$, which maps each state $s \in \mathcal{S}$ to the optimal action $a^* \in \mathcal{A}$,  
- a transition model, which determines how the environment evolves:
  - either probabilistically, $s_{i+1} \sim P(s_{i+1} \mid s_i, a_i)$,  
  - or deterministically, $s_{i+1} = f(s_i, a_i)$,  
- a loss function $\mathcal{L}(a^*, a)$, which evaluates the difference between the expert action $a^*$ and the predicted action $a$ from the learned policy.

The agent begins in an initial state $s_0$ and sequentially executes its policy to select actions. At each time step $i$, the agent selects an action according to $a_i = \pi_{\theta}(s_i)$, and transitions to the next state according to $s_{i+1} \sim P(s_{i+1} \mid s_i, a_i)$. This new state is then used to generate the next action $a_{i+1} = \pi_{\theta}(s_{i+1})$, and the process repeats.

The sequence of states and actions encountered by the agent is recorded as a trajectory $\tau = (s_0, a_0, s_1, a_1, \dots)$, which is collected through an iterative method known as the rollout algorithm. This continues until the episode is terminated.To train the policy, the loss $\mathcal{L}(a^*, a)$ is computed at each step by comparing the action $a$ taken by the learned policy against the expert action $a^*$ for the same state. This loss is then minimized through standard optimization procedures.

In imitation learning, the goal is to learn a policy $\pi_\theta(s)$ that imitates an expert policy $\pi^*(s)$ by minimizing a loss function over the space of states encountered by the learned policy.

The objective is:

$$
\arg\min_{\theta} \ \mathbb{E}_{s \sim P(s \mid \pi_\theta)} \left[ \mathcal{L} \left( \pi^*(s), \pi_\theta(s) \right) \right]
$$

- Here, $P(s \mid \pi_\theta)$ is the state distribution induced by rolling out the current learned policy $\pi_\theta$.
- $\mathcal{L}(\cdot, \cdot)$ is a task-specific loss function, such as squared error or cross-entropy.

The state distribution depends on the current policy $\pi_\theta$, meaning that the learner's own decisions affect the states it visits during training.

There are some scenarios where imitation learning is not enough in the context of self-driving. In scenarios like intersections, visual inputs alone may not provide sufficient information to decide whether to turn left, right, or go straight. To overcome these challenges, a method known as **Conditional Imitation Lerning** (CIL) was introduced [15]. Conditional Imitation Learning integrates high-level commands into the learning process: The model is trained on triplets comprising observations (e.g., images), commands (e.g., "turn left"), and corresponding expert actions. At test time, the model receives both sensory inputs and high-level commands, enabling it to make context-aware decisions.

#### 2.1) Behavior Cloning

Behavior Cloning is a special case of imitation learning where we assume access to a dataset of expert demonstrations $(s^*, a^*)$ sampled from a fixed distribution $P^*$.

The objective becomes:

$$
\arg\min_{\theta} \ \mathbb{E}_{(s^*, a^*) \sim P^*} \left[ \mathcal{L} \left( a^*, \pi_\theta(s^*) \right) \right]
$$

- $P^*$ is the expert-induced distribution over state-action pairs.
- The loss is computed by comparing the learner’s predicted action $\pi_\theta(s^*)$ to the expert action $a^*$.

Behavior cloning (BC), despite its simplicity and appeal as a supervised learning method, often fails in complex, real-world scenarios like autonomous driving. BC assumes that the learner trains on state-action pairs $(s^*, a^*)$ sampled from the expert distribution $P^*$, which implies that the model is never trained on states outside the expert's trajectory distribution $P^*$, yet at test time, small prediction errors may lead the agent to new states it hasn't seen before i.e. states outside of $P^{*}$ [16].

End-to-end autonomous driving algorithms increasingly rely on RGB images as the sole input to neural networks. However, recent studies show that multimodal perception data (i.e. combining RGB and depth) consistently outperforms unimodal RGB inputs, particularly for conditional imitation learning (CIL) agents [17]. 

### 3) Reinforcement Learning

Reinforcement learning aims to solve Markov Decision Processes (MDPs) [18]. An MDP consists of:

- a set $\mathcal{S}$ of states,  
- a set $\mathcal{A}$ of actions,  
- a transition probability function $p : \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{P}(\mathcal{S})$,  
  which to every pair $(s, a) \in \mathcal{S} \times \mathcal{A}$ assigns a probability distribution $p(\cdot \mid s, a)$, representing the probability of entering a state from state $s$ using action $a$,  
- a reward function $R : \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$,  
  which describes the reward $R(s_{t+1}, s_t, a_t)$ associated with entering state $s_{t+1}$ from state $s_t$ using action $a_t$,  
- a future discount factor $\gamma \in [0, 1]$, representing how much we care about future rewards.

The solution of an MDP is a policy $\pi : \mathcal{S} \rightarrow \mathcal{A}$ that for every $s_0 \in \mathcal{S}$ maximises:

$$
V_\pi(s_0) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_{t+1}, s_t, \pi(s_t)) \right],
$$

Where the expectation is taken over states $s_{t+1}$ sampled according to $p(s_{t+1} \mid s_t, \pi(s_t))$.

#### 3.1) Bellman Equations

One of the two Bellman equations is given by the state-value function $V_\pi(s_0)$, which is the expected cummulative discounted reward, starting from a state $s_0$:

$$
V_\pi(s_0) = \mathbb{E} \left[ R(s_1, s_0, \pi(s_0)) + \gamma V_\pi(s_1) \right].
$$

Here, the expectation is taken only over $s_1$ sampled according to $p(s_1 \mid s_0, \pi(s_0))$.

For reference, the other Bellman equation is:

$$
Q_\pi(s_0, a_0) = \mathbb{E} \left[ R(s_1, s_0, a_0) + \gamma Q_\pi(s_1, \pi(s_1)) \right].
$$

#### 3.2) Deep Deterministic Policy Gradients (DDPG)

Deep deterministic policy gradients (DDPG) [1] consists of two function approximators: a critic  $Q : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$, which estimates the value $Q(s, a)$ of the expected cumulative discounted reward upon using action $a$ in state $s$, trained to satisfy the Bellman equation:

$$
Q(s_t, a_t) = r_{t+1} + \gamma (1 - d_t) Q(s_{t+1}, \pi(s_{t+1})),
$$

under a policy given by the actor $\pi : \mathcal{S} \rightarrow \mathcal{A}$, which attempts to estimate a $Q$-optimal policy $\pi(s) = \arg\max_a Q(s, a)$.

Here $(s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1})$ is an experience tuple, a transition from state $s_t$ to $s_{t+1}$ using action $a_t$ and receiving reward $r_{t+1}$ and "done" flag $d_{t+1}$, selected from a buffer of past experiences. The error in the Bellman equality, which the critic attempts to minimise, is termed the temporal difference (TD) error.

DDPG is an **off-policy learning algorithm**, which means that the actions used during training are sampled from a policy that differs from the actor's learned optimal policy. This approach is employed to expose the learning process to a broader and more diverse range of state-action pairs than would typically be encountered under the optimal policy alone, thereby enhancing the robustness and generalization of the agent. To enhance robustness, discrete Ornstein-Uhlenbeck process noise [19] is added to the optimal policy. Therefore, at each step we add to optimal actions noise $x_t$ given by:

$$
x_{t+1} = x_t + \theta(\mu - x_t) + \sigma \epsilon_t, \tag{4}
$$

where $\theta, \mu, \sigma$ are hyperparameters and $\{\epsilon_t\}_t$ are i.i.d. random variables sampled from the normal distribution $\mathcal{N}(0, 1)$.

# Experiments

The main task to assess is that of lane following, which is the same activity evaluated by [1], [5] and [6]. This is a fundamental duty for autonomous driving and was also the task assessed by the pioneering research of Pomerleau et. al. [7]. 

The training flow is based on [6] and can be reproduced as follows:

1. Launch CARLA 0.9.14 and iterate over M episodes and T steps for each episode.

2. At the beginning of the episode, call CARLA's A* based [global_route_planner](https://github.com/carla-simulator/carla/blob/0.9.14/PythonAPI/carla/agents/navigation/global_route_planner.py) to obtain the complete
route from two points on the map. Unlike [5] and [6], training has been done using the same route in every training episode. 

3. At the beginning of each episode, an observation representing the state $s$ is constructed by concatenating the architecture-specific data entry $D$ with the driving feature vector. This results in a state representation defined as $s=([D], \phi, dt)$, which is then fed into the agent. The network outputs a set of predicted control actions $a = (\textit{throttle}, \textit{steering})$ , which are subsequently transmitted to the simulator. Based on the system’s response to these actions, a reward is computed as a function of the resulting actuation.

4. At each time step, the `lane_invasor` and `collision_sensor` are monitored. If either sensor is triggered, the current episode is terminated and immediately reset. The reset procedure involves repositioning the vehicle at the center of the lane with the correct orientation, preparing it to begin a new route. If neither sensor is activated, the training process proceeds to the next iteration.

5. Training ends when the set maximum number of episodes is met.

It is important to reflect on the rationale behind the second training point. The core idea of using the **same fixed route** across all training episodes is to deliberately overfit a neural network to that specific trajectory. This contrasts with the classical supervised learning approach to training autonomous driving agents, in which a policy is learned by fitting a neural network to a dataset of expert demonstrations. The network is trained to imitate the expert’s behavior across diverse driving scenes, with the goal of generalizing to unseen situations by minimizing prediction error on previously collected state-action pairs. In this context, however, the goal is to train a model for each route, such that the network captures the full behavioral policy required for navigating that exact path with high precision. This method draws inspiration from the fact that, in real-world applications, the route a vehicle follows during deployment often remains consistent [5]. Consequently, optimizing a policy for a fixed route may be a pragmatic and efficient approach in specific deployment scenarios such as delivery routes or structured industrial environments. Notably, this approach diverges from the methodologies adopted in [5] and [6], where the agent is trained on routes whose topologies vary with every episode. Their motivation lies in improving generalization: exposing the agent to a diverse set of route configurations encourages it to learn robust navigation strategies that transfer across environments with similar structural properties. When comparing the two paradigms, we can anticipate distinct trade-offs. The fixed-route, overfitting-based strategy is likely to yield value functions that converge more quickly, as the critic is exposed repeatedly to the same distribution of states and transitions, enabling faster policy learning. However, this comes at the cost of limited generalization. Conversely, the variable-route training paradigm may lead to slower convergence due to the broader state-action distribution, but is expected to produce agents that are significantly more adaptable and transferable to novel but structurally similar routes. This dichotomy highlights a fundamental tension in deep reinforcement learning between specialization and generalization, and suggests that the optimal choice may depend heavily on the intended deployment scenario and the variability of the environment in which the agent operates.

The results of the experimental evaluation are presented in the table below. The acronyms SLP, AN, and PER refer to Softened Lane Penalty, Action Noise, and Prioritized Experience Replay, respectively. The key observations are as follows:

- DQN 1 did not complete the route despite reproducing the [official DQN implementation](https://github.com/RobeSafe-UAH/DQN-CARLA) of DRL‑flatten‑image agent, which is reported to achieve success at episode 16500 out of 20000 [6]. This discrepancy is most likely attributable to the current method of image preprocessing.

- DDPG 1 also did not complete the route, despite reproducing the [official DDPG implementation](https://github.com/RobeSafe-UAH/DQN-CARLA) of DRL‑flatten‑image agent, and reported to be completed at episode 50/500 [6]. The failure is most plausibly due to the overly strict enforcement of the lane invasion penalty.

- DDPG 2 demonstrates rapid learning of lane-following behavior, at episode 41. However, it consistently fails to avoid lane invasion on the second curve (as illustrated in the video below). Notably, it does not exhibit any capacity for self-correction after 10000 episodes.

- DDPG 3 reveals that simply introducing a Softened Lane Penalty (SLP) enables the agent to follow the lane effectively without incurring lane invasions at episode 609 (as illustrated in the video below).

- DDPG 4 introduces action-level exploration through Ornstein–Uhlenbeck (OU) noise [19] and requires a greater number of episodes to complete the route compared to the SLP-only agent. This outcome is expected, as increased exploration leads the agent to sample a wider range of actions. While this degrades short-term performance, it promotes improved robustness over longer horizons.

- DDPG 5 integrates Prioritized Experience Replay (PER) in an effort to enhance training efficiency. However, a comparison of convergence, 1559 episodes (OU + PER) versus 991 episodes (OU only), suggests that PER may be impeding convergence. This could result from suboptimal prioritization signals or a mismatch between prioritization and the exploratory nature of the collected data.

- DDPG 6 extends DDPG 5 by modifying the actor and critic architectures to replicate exactly those proposed by Lillicrap et al. [1]. Surprisingly, this configuration enables the agent to complete the route more quickly than DDPG 5, albeit with lane invasions (as shown in the video below). This result indicates strong potential. However, the agent still fails to learn corrective behavior after 10000 episodes, highlighting the need for further fine-tuning of the revised network architectures.


The network architecture for both DQN and DDPG is the same that [6] except for DDPG 6, which follows [1]'s. For DQN 1, the input state vector is composed by the flattened B/W segmented image data of 121 components, which is concatenated with the distance to the center of the lane $d_t$ and the angle between the vehicle and the center of the lane $\phi_t$ i.e. $s = ([p_{t0}, p_{t2}, ..., p_{t120}], \phi_t, d_t)$. For DQN 2 and all DDPG agents in this project, the waypoints are retrieved directly from the CARLA simulator using the available Python API. These waypoints are provided in global coordinates, referenced to the origin point $(0, 0, 0)$ of CARLA’s map. As a result, feeding them directly into the MLP would be incorrect. To make them meaningful for the agent, the waypoints must first be transformed into the coordinate frame of the ego vehicle. This is achieved by applying a transformation matrix that includes both rotation and translation, using the vehicle’s global position $[X_c, Y_c, Z_c]$ and current heading (yaw angle) $\phi_c$:


$$
T = \begin{bmatrix}
\cos \phi_c & -\sin \phi_c & 0 & X_c \\
\sin \phi_c &  \cos \phi_c & 0 & Y_c \\
0           & 0            & 1 & Z_c \\
0           & 0            & 0 & 1
\end{bmatrix}
$$


Empirically, the input is fixed to a window of 15 waypoints [6]. At each time step, this list is updated in a First-In-First-Out (FIFO) manner, starting from the waypoint closest to the vehicle and extending to the next 14 along the planned route. The resulting 15 local waypoints, each with $(x, y)$ coordinates, are concatenated with the vehicle's current lateral displacement $dt$ and heading angle $\phi_t$ yielding the final input state vector: $s = ([wp_{t0}, wp_{t2}, ..., wp_{t14}], \phi_t, d_t)$. Each component of this waypoint list forming the State vector has coordinates $(x, y)$. The models are trained using only the x-coordinates. This choice is motivated by the fact that the lateral component (the x-position relative to the vehicle’s frame) contains the most critical information for lane-keeping tasks [6].



| Agent | Network input | Configuration | Training Episodes | Best Episode | Route Completion |
| - | - | - | - | - | - |
| DQN 1           | B/W Image            | Default  | 20000    |  -   | NO |
| DQN 2           | Waypoints    | Default  | 20000  |  5616    | YES |
| DDPG 1 | Waypoints    | Default  |  10000  | -  |  NO   |
| DDPG 2 | Waypoints    | No lane penalty  |  10000  | 41  |  YES  |
| DDPG 3            | Waypoints | SLP |  10000  | 609  |  YES   |
| DDPG 4 | Waypoints | SLP + AN  |  10000  | 991  |  YES   |
| DDPG 5               | Waypoints    | SLP + AN + PER  |  10000   | 1559  |  YES  |
| DDPG 6                | Waypoints    | SLP + AN + PER + [1]  |  10000  | 243  |  YES   |


# Videos 

[Showcase of DDPG 2 (Episode 41)](https://github.com/your-user/your-repo/issues/1)

[Showcase of DDPG 3 (Episode 609)](https://github.com/your-user/your-repo/issues/1)

[Showcase of DDPG 6 (Episode 243)](https://github.com/your-user/your-repo/issues/1)


# Discussion & Future Work

These results could potentially be improved by designing more informative state representations. One widely used approach is the Variational Autoencoder (VAE) [20], which has been shown to significantly accelerate the learning process of DDPG agents deployed on real vehicles. However, the same benefit is not observed when the agent is trained in simulation environments [5], likely due to differences in sensory noise, dynamics, or task realism. This highlights the need for state representations that generalize across both simulated and real domains. While pixel-based autoencoders are a starting point, the broader field of computer vision offers a rich set of tools for extracting semantically meaningful features. Techniques such as semantic segmentation, depth estimation, egomotion prediction and optical flow have demonstrated strong priors about what visual information is crucial for driving tasks [21]. These structured representations can encode high-level cues (like lane markings, obstacles and motion patterns) far more explicitly than raw pixels or latent vectors. To unlock their full potential, such perceptual insights must be systematically integrated into reinforcement learning pipelines (whether model-free or model-based) so agents can leverage abstract visual understanding to make more robust and efficient decisions in complex, real-world scenarios.

However, unsupervised state encoding methods alone will likely not be enough for efficient policy learning, especially in environments where sample efficiency is critical. To compress the state representation in a way that facilitates rapid policy learning with minimal data, the encoding must capture not just generic features of the observation, but specifically those elements that are relevant to decision-making and long-term outcomes. Identifying which aspects of the image observations are important requires guidance from the reward and control signals, as these provide the only cues about what ultimately matters for the task. Integrating such task-relevant supervision into the encoding process can be done through various mechanisms, such as jointly training the encoder with the policy or using auxiliary tasks that reflect reward structure. However, an important challenge in this process is the problem of credit assignment (rewards observed at a particular time step may depend on states or actions taken many steps earlier). This temporal disconnect means that effective state encoding must include a temporal modeling component, such as recurrent architectures, attention mechanisms, or predictive world models [22], in order to propagate reward information back through time and correctly attribute relevance to earlier observations. Without this, the encoder risks focusing on visually salient but task-irrelevant features, limiting policy performance and generalization.

New advances in model-based reinforcement learning provide exciting alternative avenues for autonomous driving research, offering the potential to move beyond the limitations of model-free approaches that rely heavily on dense interaction with the environment. One particularly promising direction is exemplified by MILE (Model-based Imitation LEarning) [22], which demonstrates how learning to **simulate the world and plan in imagination** can significantly enhance policy robustness and generalization. MILE jointly learns both a predictive model of the environment and an imitation-based driving policy, trained entirely offline using high-resolution videos of expert demonstrations (without requiring any online environment interaction). Crucially, it incorporates 3D geometry as an inductive bias (i.e. the model is explicitly designed to understand and organize the world in terms of 3D spatial structure, even though it only sees 2D camera images), enabling the construction of a compact latent representation that encodes static scene layout, dynamic agents, and ego-behavior in a unified manner. The model can simulate diverse and plausible driving trajectories and decode them into interpretable bird’s-eye view (BEV) semantic segmentations. Notably, MILE is the first camera-only approach capable of modeling static and dynamic elements of urban environments together with ego-motion, and it shows strong generalization to new conditions. Based on CARLA, MILE outperforms prior state-of-the-art methods in driving score when deployed in an entirely new town under novel weather conditions, demonstrating the power of learning to act through imagination. Its ability to generate complete plans internally (without querying the environment) positions it as a groundbreaking step toward scalable, data-efficient and generalizable autonomous driving systems.

The MILE world model is trained using supervision from ground-truth labels, which helps accelerate convergence and stabilize learning. While such labels are readily available in simulated environments like CARLA, acquiring high-quality labels in the real world (such as dense semantic segmentation, depth maps, or motion trajectories) remains expensive, labor-intensive and often impractical at scale. A promising alternative lies in **self-supervised learning**, which eliminates the need for manual labeling by relying on intrinsic signals derived from the data itself. One such approach is to use image reconstruction as the training objective, where the model learns to predict future frames or viewpoints based on current observations. By integrating geometric constraints such as depth and scene flow into this framework, the model can be guided to learn more structured and physically consistent representations [23]. Specifically, if the latent representation at time step $t$ can be used to infer the depth map of the current image, the ego-pose transformation from $t-1$ to $t$, and the residual 3D scene flow (i.e. motion of dynamic objects relative to the background), then a reprojection loss can be applied. This loss enforces not only accurate depth prediction but also precise modeling of ego-motion and the motion of surrounding agents. Additionally, scale ambiguity (a common issue in monocular self-supervision) can be addressed by anchoring the estimated ego-pose using ground-truth vehicle velocity [24]. Combining these ideas could make self-supervised world models viable in real-world driving scenarios, reducing reliance on simulation and enabling continual learning from real-world deployment.

Building on the promise of self-supervised learning for real-world driving, an even more ambitious direction is the development of **foundation vision models**: large-scale neural networks trained on vast, unlabeled datasets with the goal of learning general-purpose visual representations. These models mirror the success of large language models (LLMs) like GPT-4 [25], which was trained on internet-scale text data to predict the next word, and can solve a wide range of downstream tasks (e.g., question answering, code generation, translation...) even in a zero-shot setting by simply conditioning on a task-specific prompt. In vision, the goal is similar: to train a single, unified model on millions of videos so that it implicitly learns geometry, motion, and semantics in a way that can be adapted to diverse tasks such as 3D object detection, video panoptic segmentation, depth and scene flow prediction, or future frame generation. In this sense, such a model could also function as a world model, capable of inferring latent dynamics that explain how an agent’s actions influence its environment over time. A world model that can reconstruct an agent’s sensory experience (particularly in dynamic, ego-centric scenarios like autonomous driving) must inherently learn the underlying structure of 3D scenes, object interactions, and temporal dependencies. The resulting latent representations could be fine-tuned for multiple computer vision and control tasks, offering a scalable alternative to hand-crafted modules. However, realizing this vision comes with formidable challenges. The first is data: while online video data is abundant, most of it involves passive observation (e.g. static cameras), which lacks the interactive component critical for understanding causality and agent-environment dynamics. Datasets consisting of ego-centric driving videos, where the agent interacts with and alters its environment, are essential to ground the model in real-world physics and control. Secondly, model architecture and compute remain key bottlenecks. Like in NLP, achieving strong generalization in vision will likely require models with hundreds of billions of parameters, pushing the limits of current hardware. Transformers [26], which already dominate language modeling and are replacing convolutional architectures in vision, are strong candidates for this scale due to their token-based structure and scalability. They are beginning to show promise in video modeling as well, although video input presents greater memory and compute demands than text. Overcoming these challenges will require not only architectural advances but also breakthroughs in optimization, gradient storage and hardware acceleration. Still, the reward is enormous: a successful foundation vision model would mark a profound step forward in **embodied intelligence**, enabling systems to perceive and act in complex, unstructured environments with human-like adaptability and unlocking a wide range of capabilities far beyond today’s modular or task-specific perception systems.



# References

1. T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” in International Conference on Learning Representations (ICLR), 2016.

2. T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” in International Conference on Learning Representations (ICLR), 2015.

3. A. Dosovitskiy, G. Ros, F. Codevilla, A. M. López, and V. Koltun. “CARLA: An open urban driving simulator,”. in Conference on Robot Learning (CoRL), 2017.

4. V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, p. 529, 2015.

5. A. Kendall, J. Hawke, D. Janz, P. Mazur, D. Reda, J. M. Allen, V. D. Lam, A. Bewley, and A. Shah, “Learning to drive in a day,” in International Conference on Robotics and Automation (ICRA), 2019.

6. O. Pérez‑Gil, R. Barea, E. López‑Guillén, L. M. Bergasa, C. Gómez‑Huélamo, R. Gutiérrez, and A. Díaz‑Díaz, “Deep reinforcement learning based control for Autonomous Vehicles in CARLA,” Springer, Multimedia Tools and Applications 2022.

7. M. Montemerlo, J. Becker, S. Bhat, H. Dahlkamp, D. Dolgov, S. Ettinger, D. Haehnel, T. Hilden, G. Hoffmann, B. Huhnke, et al., “Junior: The stanford entry in the urban challenge,” Journal of field Robotics, vol. 25, no. 9, pp. 569–597, 2008.

8. S. Thrun, W. Burgard, and D. Fox, Probabilistic robotics. MIT press, 2005.

9. C. Linegar, W. Churchill, and P. Newman, “Made to measure: Bespoke landmarks for 24-hour, all-weather localisation with a camera,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), Stockholm, Sweden, May 2016.

10. P. F. Alcantarilla, S. Stent, G. Ros, R. Arroyo, R. Gherardi, “Street-view change detection with deconvolutional networks,” Springer, Autonomous Robots, 2018.

11. V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for scene segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

12. G. Ros, L. Sellart, J. Materzynska, D. Vazquez, and A. M. López, “The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 3234-3243. 

13. G. Ros, S. Ramos, M. Granados, A. Bakhtiary, D. Vazquez, and A. M. López “Vision-Based Offline-Online Perception Paradigm for Autonomous Driving,” IEEE Workshop on Applications of Computer Vision (WACV), 2015.

14. D. A. Pomerleau, “Alvinn: An autonomous land vehicle in a neural network,” in Advances in neural information processing systems, 1989, pp. 305–313. 

15. F. Codevilla, M. Müller, A. M. López, V. Koltun, and A. Dosovitskiy, “End-to-end driving via conditional imitation learning,” IEEE International Conference on Robotics and Automation (ICRA), 2018.

16. F. Codevilla, E. Santana, A. M. López, and A. Gaidon, “Exploring the Limitations of Behavior Cloning for Autonomous Driving,” Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9329-9338.

17. Y. Xiao, F. Codevilla, A. Gurram, O. Urfalioglu, and A. M. López, “Multimodal End-to-End Autonomous Driving,” IEEE Transactions on Intelligent Transportation Systems, 2020.

18. R. S. Sutton, A. G. Barto, et al., Reinforcement learning: An introduction. MIT press, 1998.

19. G. E. Uhlenbeck and L. S. Ornstein, “On the theory of the brownian motion,” Phys. Rev., vol. 36, pp. 823–841, Sep 1930.

20. D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in The International Conference on Learning Representations (ICLR), 2014.

21. A. Kendall, Y. Gal, and R. Cipolla, “Multi-task learning using uncertainty to weigh losses for scene geometry and semantics,”. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

22. A. Hu, G. Corrado, N. Griffiths, Z. Murez, C. Gurau, H. Yeo, A. Kendall, R. Cipolla, and J. Shotton, “Model-Based Imitation Learning for Urban Driving,”. Advances in Neural Information Processing Systems, 2022.

23. J. Hur, and S. Roth, “Self-Supervised Monocular Scene Flow Estimation,”. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

24. V. Guizilini, R. Ambrus, S. Pillai, A. Raventos, and A. Gaidon, “3D Packing for Self-Supervised Monocular Depth Estimation,”. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2485–2494.

25. OpenAI, “GPT-4 Technical Report,” 2024

26. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is All you Need,”. Advances in Neural Information Processing Systems (NIPS), 2017.














