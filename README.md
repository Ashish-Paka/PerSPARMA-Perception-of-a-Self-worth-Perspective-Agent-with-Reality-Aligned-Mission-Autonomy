<div align="center">

# PerSPARAMA

**Perception of a Self worth Perspective Agent with Reality Aligned Mission Autonomy**

Continual Self-Supervised World and Agent Modeling via Sparse Transformer Perception and Action-Chunked Reinforcement Learning for Risk-Calibrated Autonomy in Unstructured Environments

---

[![Status](https://img.shields.io/badge/Status-Working%20Draft-yellow)](.)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Ashish Paka** | MS Robotics, LOGOS Robotics Lab, Arizona State University

*A robot that knows it is tired, learns from what happens to it, and cares enough about itself to survive — will make better decisions than a robot that does not.*

</div>

---

## Abstract

Autonomous agents deployed in GPS-denied, communication-degraded environments face a compound failure mode: the agent's world model and self-model degrade simultaneously, silently, and in ways that corrupt the diagnostic mechanisms that would normally detect degradation. Existing approaches treat perception, self-monitoring, and decision-making as separable subsystems — producing agents that are capable within their characterized domain but brittle at its boundary and incapable of growth beyond it.

Presenting **PerSPARAMA**, a unified architecture coupling a sparse voxel transformer world model with a continuously updated homeostatic self-model through bidirectional cross-attention (`Z_relation`), such that every perceptual act is conditioned on the perceiver's physical state and every self-evaluation is conditioned on the current scene. Drawing on the neuroscience of discrete temporal integration — the ~400ms conscious ignition window of Global Workspace Theory — as a biological design prior, the architecture implements a buffer-commit-correct action cycle with adaptive temporal commitment. Skill boundary detection is generalized from named skill invocation to every decision the agent makes. Action selection uses an action-chunked transformer governed by dual-utility PPO, where mission reward and self-preservation reward are arbitrated through a learned, state-dependent exchange rate `α(t)` — operationalizing self-worth as a continuous prior rather than a safety constraint. The architecture is embodiment-agnostic via a formal interface contract.

**Keywords:** homeostatic self-modeling, dual-utility RL, action chunking, sparse voxel transformers, continual learning, self-worth, OOD detection, embodied autonomy, skill boundary detection

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Related Work and Gap Analysis](#2-related-work-and-gap-analysis)
- [3. Problem Formulation](#3-problem-formulation)
- [4. Architecture](#4-architecture)
- [5. Implementation](#5-implementation)
- [6. Case Study](#6-case-study)
- [7. Discussion](#7-discussion)
- [8. Limitations](#8-limitations)
- [9. Conclusion](#9-conclusion)
- [References](#references)
- [Appendix](#appendix)

---

## 1. Introduction

### 1.1 Motivation

Consider a wheeled robot at hour five of a six-hour solo cave mission: battery at 38%, wheel encoders slipping on wet limestone for ninety minutes, the map from hour one now four hours stale. It encounters a steep downward slope toward a high-value target. Every existing system evaluates this against a static robot model and a world model with no temporal uncertainty. Both are wrong, in ways the system cannot detect from within.

The failure is structural: **the agent cannot externally validate its own state**, yet must make irreversible decisions conditioned on the accuracy of that state. Current autonomous architectures do not maintain a living model of who they currently are, hold no intrinsic stake in their own continuity, and do not grow through accumulated experience. PerSPARAMA is a direct architectural response to this failure class, derived from three governing principles.

### 1.2 Philosophical Foundation

#### 1.2.1 Reality, Causality, and the Emergent Agent

All agents exist inside a physical causal chain they cannot break. Yet an agent architecturally biased toward honest model updating — one that treats sensor data as the closest proxy for physical truth and revises rather than defends — produces different outputs through the same causal chain than one that does not. The effort to align with reality is itself a causal force that shifts the probability distribution over future states.

This is the precise meaning of **hope** in this work:

> *There is no real choice that changes our future, but we can hope to change the probability of our future being so deterministic by putting our morality and our acceptance of reality on this side of the emergent side of things.*

Hope is the act of a deterministic system loading itself with honest models and genuine values before the physics runs forward. **Architectural instantiation:** the reality alignment loss `L_reality` and the structural bias toward revision over defense throughout all three modules.

#### 1.2.2 Self-Worth and the Valuation of Continuity

Standard RL produces agents indifferent to their own continuity — an agent will destroy itself if that maximizes cumulative reward. This indifference produces brittle, catastrophically irreversible behavior in novel environments. Self-worth theory proposes that an agent maintain an internal model of its own value: *I am a system with a particular configuration, accumulated capability, and remaining potential, and the preservation of that has weight in my decisions.*

Three observations ground this:
1. A robot operating for five hours under stress is **not the same robot** that was deployed
2. A path acceptable at deployment may represent **unacceptable cumulative loading** on the current robot
3. Self-preservation and mission completion have **no fixed exchange rate** — the ratio is state-dependent

**Architectural instantiation:** the dual-utility reward `R_total = α(t)·R_mission + (1−α(t))·R_self + γ·R_universal`, the learned exchange rate `α(t)`, and the homeostatic state vector `H(t)`.

#### 1.2.3 The Growing Agent: Developmental Architecture

A child grows rather than merely operates through two parallel processes: (1) **reality alignment** — continuous updating of the world model against physical fact, and (2) **self-model accumulation** — a living record of capabilities, limits, and their dependence on physical condition. The robot must run the same loop. The skill absent at deployment exists at hour six because the loop encountered the right situations. Risk calibration naive at hour one is honest at hour ten because physics kept correcting it.

**Architectural instantiation:** competence memory `C` conditioned on `H(t)`, the skill boundary detector (execute / adapt / improvise), and the reality alignment loop that updates both world and self simultaneously.

### 1.3 Biological Parallel: The 400ms Snapshot

Neuroscience of consciousness reveals that biological perception is not continuous but a series of discrete ~400ms integration windows:

1. **Unconscious buffer (0–400ms):** Analog, parallel, probabilistic sensory accumulation — the brain computes the most likely reality from noisy transduction without committing to an interpretation.
2. **Integration snapshot:** Features bind into a discrete packet that "ignites" across the Global Workspace, transitioning from unconscious analog processing to conscious symbolic representation.
3. **Postdiction:** The brain back-dates the experience to stimulus onset, producing subjective immediacy despite ~400ms integration latency.

| Biological Phase | PerSPARAMA Analog | Function |
|---|---|---|
| Unconscious buffer | Continuous multi-modal sensor fusion (10–20 Hz) | Probabilistic accumulation before commitment |
| Integration snapshot | Discrete action chunk commitment | Binding of percept + self-state into behavioral unit |
| Postdiction | Reality alignment loss `L_reality` | Backward-looking correction updating all models |

The ~400ms window is an evolutionary optimum trading noise resolution against reaction latency. PerSPARAMA's adaptive chunk length instantiates this tradeoff computationally, modulated by `ε_ood` and `H(t)` rather than fixed by evolution. PerSPARAMA does not claim consciousness — it claims that the temporal structure **buffer, commit, correct** is a design prior worth implementing.

### 1.4 Core Thesis

Any situated agent exists at the intersection of three irreducible components: (1) **Perception** — the only access to reality, always provisional; (2) **Self** — who the agent is right now, continuously updated; (3) **Mission** — the attractor giving decisions direction and a morality floor. These cannot be treated as separable subsystems. The world must be perceived through the self. The self must grow from what the world teaches it. The mission must be pursued in proportion to what the self can honestly afford.

Computationally: `Z_world` conditioned on `Z_self` via self-as-token, dual-utility `[R_mission, R_self]` arbitrated by learned `α(t)`, and competence memory accumulating entries tagged with physical state at time of experience.

### 1.5 Contributions

1. **Self-as-token in voxel attention** — `H(t)` embedded as a learned token in the sparse voxel transformer, conditioning scene representation on the perceiver's physical state
2. **Bidirectional cross-attention perspective** — `Z_relation` via world-queries-self and self-queries-world, producing a joint situational latent
3. **Dual-utility RL with learned exchange rate** — `α(t)` as a learned function of joint physical-mission state, producing emergent risk calibration
4. **State-conditioned competence memory** — skill retrieval weighted by both situational similarity and physical state compatibility
5. **Generalized skill boundary detection** — three-way classification (execute / adapt / improvise) applied to every decision, not only named skills
6. **Temporal voxel state with learned decay** — per-voxel age, modality provenance, and environment-class-dependent decay rate
7. **Embodiment interface contract** — formal abstract interface enabling identical cognitive architecture across wheeled, aerial, aquatic, and humanoid platforms

---

## 2. Related Work and Gap Analysis

### 2.1 Prior Work

**Sparse voxel perception.** VoxFormer [1] introduced sparse query-based 3D semantic scene completion via deformable cross-attention. Act3D [2] and RISE [3] extended 3D feature fields for manipulation. PerSPARAMA extends VoxFormer with multi-modal provenance-tracked seeding, temporal voxel state, and self-as-token injection.

**Action-chunked policy learning.** ACT [4] addressed compounding error via CVAE-based action sequence prediction. PerSPARAMA extends ACT with adaptive chunk length modulated by `ε_ood` and `min(H(t))`, plus dual-head value estimation.

**Continual skill learning.** ACT-LoRA [5] achieves 100% novel skill success with 74.75% pre-trained retention (vs. 0.25% for TAIL [6]). Its queryable skill library is the direct parent of PerSPARAMA's skill boundary detector, generalized from named skills to every decision. LOTUS [7] and XSkill [8] contribute unsupervised discovery and cross-embodiment alignment.

**World models.** Surveys [9, 10] taxonomize embodied world models across functionality, temporal modeling, and spatial representation. PerSPARAMA occupies the decision-coupled, sequential-simulation, spatial-grid cell — uniquely modeling the perceiving agent as a token in its own representation.

**Continual learning.** Sirius [11] demonstrated human-in-the-loop continual deployment learning. CLP [12] addressed unsupervised novelty detection. PerSPARAMA eliminates the human feedback requirement: `L_reality` is the physics-provided supervisor.

**OOD detection.** Task-driven OOD with PAC guarantees [13], GMM-based LiDAR uncertainty [14], transition-based RL OOD guarantees [15], and system-level OOD framing [16] ground PerSPARAMA's composite `ε_ood`.

**Safety.** R²AI [24] proposes intrinsic self-sustaining safety objectives. PerSPARAMA operationalizes this: self-preservation is a coequal utility, not a constraint.

### 2.2 Gap Analysis

| Contribution | Prior Work | What PerSPARAMA Adds |
|---|---|---|
| Self-as-token in voxel attention | VoxFormer [1], Act3D [2], RISE [3] | No prior work conditions scene perception on physical state as a token |
| Temporal voxel state with decay | VoxFormer [1] | No prior work models per-voxel age with environment-class-dependent decay |
| `σ_kinematic` cross-modal inference | IMU/odometry fusion lit. | No prior work infers kinematic model *validity* vs. error *magnitude* |
| Dual utility with learned `α(t)` | Multi-objective RL lit. | No prior work makes the exchange rate a learned function of joint state |
| State-conditioned competence memory | RT-1 [19], LOTUS [7], XSkill [8] | No prior work conditions skill retrieval on current physical state |
| Generalized skill boundary | ACT-LoRA [5] | Generalized from named skills to every path, sensor, and timing decision |
| Joint world+self reality alignment | Sirius [11], continual learning lit. | Prior work updates world model only; PerSPARAMA updates self simultaneously |
| Embodiment interface contract | Cross-embodiment lit. | No formal interface for identical cognitive architecture across embodiments |

Existing autonomous architectures typically exhibit one or more of four limitations: **(A)** strong perception but no self-model, **(B)** a self-model that is static (deployment-time characterization), **(C)** learning only from external supervision, or **(D)** safety treated as a hard constraint rather than a value. PerSPARAMA addresses all four simultaneously and couples them through a shared differentiable latent state.

---

## 3. Problem Formulation

### 3.1 Unified Problem Statement

> *How can an autonomous agent, operating without external supervision in GPS-denied and communication-denied environments, maintain a continuously self-correcting model of both its external environment and its own physical and epistemic state — such that its decisions are conditioned not on a static deployment-time characterization, but on an accumulated, fact-grounded representation of what it currently is, what it has learned, and what it can honestly claim to know?*

### 3.2 Problem Statements

Seven faces of a single condition: **a solo agent must make irreversible decisions under uncertainty it cannot externally validate.**

| ID | Problem | Core Failure Mode |
|---|---|---|
| **PS-1** | Second-Order Uncertainty in GPS-Denied Localization | Confidence bounds wrong, not just large; error model itself degrades silently |
| **PS-2** | Resource-Aware Irreversible Commitment | Return cost unknown at commitment time; robot and map simultaneously changing |
| **PS-3** | Kinematic Model Validity Across Heterogeneous Terrain | Surface transition invalidates odometry model undetectably |
| **PS-4** | Homeostatic State for Path Evaluation | Traversability evaluated against static deployment model |
| **PS-5** | Dual Utility Arbitration | No fixed mission/self-preservation exchange rate exists |
| **PS-6** | Retroactive World Model Invalidation | Sensor degrades at T−Δ, detected at T; observations in [T−Δ, T] embedded with inflated confidence |
| **PS-7** | Epistemic Humility in Novel Environments | Agent cannot detect it has exceeded competence boundary |

```
PS-1, PS-3  →  Uncertainty estimates WRONG, not merely large
PS-2, PS-6  →  Irreversible decisions on INCOMPLETE information
PS-4, PS-5  →  Self-model QUIETLY INVALID through degradation
PS-7        →  Developmental loop BREAKS outside characterized domain
```

### 3.3 Reward Structure

The reward is a **vector**, not a scalar — `R_mission` and `R_self` are evaluated independently and combined through a learned exchange rate, never collapsed prematurely:

```
R_total = α(t) · R_mission + (1 − α(t)) · R_self + γ · R_universal

R_mission   = 0.4·coverage_Δ + 0.2·data_quality + 0.3·objective_progress + 0.1·return_feasibility
R_self      = 0.4·H_t_maintained + 0.3·capability_preserved + 0.3·stress_minimized
R_universal = −λ₁·irreversible_env_harm − λ₂·unnecessary_self_harm

α(t) = ExchangeRateNetwork(H_t, mission_progress, resource_remaining, ε_ood)
γ > 0  [permanent floor]
```

Note: `return_feasibility` in `R_mission` is intentionally self-referential — even the mission objective accounts for the agent's ability to return, coupling mission progress to self-preservation at the reward level before `α(t)` arbitration.

---

## 4. Architecture

### 4.1 System Overview

<div align="center">
<img src="assets/persparama_architecture.svg" alt="PerSPARAMA Architecture Flowchart" width="680"/>
<br/>
<sub><b>Figure 1.</b> PerSPARAMA architecture — from physical reality through sensor fusion, world and self modules, bidirectional cross-attention (Z_relation), skill boundary detection, dual-utility decision, action chunking, embodiment, and reality alignment feedback loop.</sub>
</div>

<br/>

<details>
<summary><b>Text-based architecture diagram</b> (for terminal / plain-text readers)</summary>

```
┌──────────────────────────────────────────────────────────────┐
│                    SHARED LATENT STATE                        │
│           Z = [Z_world ‖ Z_self ‖ Z_relation]                │
└──────────────┬──────────────────────┬────────────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌────────▼───────────────────────┐
    │   WORLD MODULE       │  │        SELF MODULE              │
    │   VoxFormer-derived  │  │  HomeostaticEncoder H(t)        │
    │   Sparse Voxel       │◄─►│  CompetenceMemory C            │
    │   Transformer        │  │  OODDetector ε_ood              │
    └──────────┬──────────┘  └────────┬───────────────────────┘
               │  bidirectional        │
               │  cross-attention      │
               └──────────┬────────────┘
                          │  Z_relation
               ┌──────────▼──────────────────────────────────┐
               │           DECISION MODULE                     │
               │  SkillBoundaryDetector (Δ)                   │
               │  ACTTransformer (adaptive chunk length)       │
               │  DualUtilityPPO [R_mission, R_self]          │
               │  ExchangeRateNetwork α(t)                     │
               └──────────┬──────────────────────────────────┘
                          │ action chunk
               ┌──────────▼──────────────────────────────────┐
               │     EMBODIMENT ABSTRACTION LAYER             │
               │  Wheeled · Aerial · Aquatic · Humanoid        │
               └──────────┬──────────────────────────────────┘
                          │
                          └─── new perception → loop
```

</details>

Three co-running modules share a structured latent state; no module is statically upstream in steady state. All modules are structurally biased toward revision over defense (Section 1.2.1).

### 4.2 World Module

**Base:** VoxFormer [1], extended with three components.

**Multi-modal voxel seeding.** Queries seeded from LiDAR (geometric), stereo (semantic), ultrasonic (near-field), and IMU (orientation). Each query carries **seed provenance** — enabling downstream uncertainty attribution per map region.

**Temporal voxel state.** Each voxel carries semantic class, occupancy probability, observation timestamps, modality history, and a learned per-environment decay rate:
```
confidence(T_now) = occupancy_prob × exp(−decay_rate × age)
```
A passage mapped at T=0 carries age-discounted confidence at T=4h. The decay rate is updated by the continual learning loop. Addresses **PS-4**. Combined with `σ_kinematic` and `η_sensor`, this produces **second-order uncertainty** — confidence in confidence — because region confidence is discounted by the health of the sensors that built it:
```
effective_confidence(region) = confidence(region) × health(seed_modalities(region))
```
This addresses **PS-1**: the agent maintains not just a position estimate with uncertainty bounds, but a meta-estimate of how trustworthy those bounds are, derived from cross-modal consistency rather than the corrupted model itself.

**Self-as-token.** `H(t)` is embedded as a learned token `q_self` in the voxel transformer's attention sequence:
```
Q = [q_voxel_1, ..., q_voxel_N, q_self]
Z_world = SparseDeformableAttention(Q, K, V)
q_self = MLP([T_thermal ‖ I_wheel ‖ V_battery ‖ σ_kinematic ‖ η_sensor ‖ τ_stress])
```
`Z_world` is therefore conditioned on the perceiver's physical state — the same geometry produces a different latent for a fresh versus degraded robot. This mirrors biological perception: integration is always conditioned on the integrator's physiological state.

### 4.3 Self Module

**Homeostatic state `H(t) ∈ [0,1]⁶`.** Continuous, non-thresholded, published at 20 Hz:

| Component | Measures | Source |
|---|---|---|
| `T_thermal` | Motor + compute thermal load | Thermistors |
| `I_wheel` | Wheel current draw history | Motor driver |
| `V_battery` | Remaining power fraction | BMS |
| `σ_kinematic` | Kinematic model validity | Cross-modal inference |
| `η_sensor` | Sensor health composite | Self-diagnostics |
| `τ_stress` | Accumulated mechanical stress | Integrated load history |

No mode switch. Behavior shifts fluidly as `H(t)` degrades — analogous to human decision-making under fatigue.

**`σ_kinematic`** is inferred, not measured, from cross-modal consistency between encoder, visual, and IMU odometry plus surface class and slip history. When sources disagree beyond a learned threshold, `σ_kinematic` degrades — detecting kinematic model *invalidity* as distinct from odometry *error*. Addresses **PS-3**.

**Competence memory `C`.** FAISS-indexed key-value store. Each entry: `(terrain_embedding ∈ ℝ¹²⁸, action_embedding ∈ ℝ⁶⁴, outcome_score ∈ [0,1], robot_state_H_t ∈ ℝ⁶, environment_class, mission_phase, is_improvised)`. Retrieval weighted by situational similarity **×** physical state compatibility — the agent learns what it can do *in what condition*.

**OOD detector `ε_ood ∈ [0,1]`.** Composite of voxel semantic statistics, kinematic anomaly signal, cross-modal consistency, and competence memory distance. When `ε_ood` is elevated, three responses activate proportionally: (1) global confidence discount on all planning outputs, (2) planning horizon contraction via shorter action chunks, and (3) increased self-monitoring frequency:
```
chunk_length     = max(min_chunk, base_chunk × (1 − λ_ood · ε_ood))
monitor_interval = base_interval / (1 + κ · ε_ood)
```
Addresses **PS-7**.

**Retroactive uncertainty revision (PS-6).** When `η_sensor` drops (sensor degradation detected at time T), the system estimates a degradation onset window [T−Δ, T] from the sensor's consistency history. Map regions built during that window are identified via seed provenance tracking, and their confidence is retroactively inflated by the degradation factor:
```
revised_confidence(region) = original_confidence × (η_sensor_at_build_time / η_sensor_assumed)
```
This prevents degraded-sensor observations from persisting as high-confidence foundational structure in the world model.

### 4.4 Relation Module

Bidirectional cross-attention producing **perspective** — the situation as experienced by this agent in this state:
```
Z_w2s = CrossAttention(Q=Z_world, K=Z_self,  V=Z_self)   # world queries self
Z_s2w = CrossAttention(Q=Z_self,  K=Z_world, V=Z_world)  # self queries world
Z_relation = LayerNorm(Z_w2s + Z_s2w)
```
A degraded and a fresh robot produce different `Z_relation` from identical observations.

### 4.5 Decision Module

**Skill boundary detection.** Before action selection:
```
Δ, nearest_skill = C.nearest(SituationEncoder(Z_relation), conditioned_on=H(t))

if   Δ < θ_known  → load LoRA_adapter[nearest_skill.id]; execute
elif Δ < θ_novel  → initialize new LoRA adapter from nearest; adapt online
else              → improvise (below); write outcome to C regardless of success
```
Each skill in `C` is associated with a LoRA adapter (rank-8), stored by `skill_id`. Known skill execution loads the corresponding adapter; adaptation initializes a new adapter from the nearest skill's weights. Derived from ACT-LoRA [5], generalized to run on *every* decision — path selection, sensor allocation, return timing, terrain commitment.

**Improvisation.** When `Δ > θ_novel`, the agent faces a genuinely novel situation. Improvisation is not stochastic exploration — it is the base policy (frozen pretrained weights) producing an action chunk from the novel `Z_relation`, which itself encodes the agent's full situational and self-state context via cross-attention. The outcome — regardless of success or failure — is written to `C` with `is_improvised=True`, and a new rank-8 LoRA adapter is initialized from the resulting trajectory for future retrieval.

**Adaptive action chunking (ACT [4]).** Predicts `k`-step action sequences — commitment to a short behavioral unit before full reasoning completes (analogous to intuition — the fast, pre-deliberative commitment to a course of action):
```
chunk_length = max(2, ⌊base × (1 − 0.6·ε_ood) × min(H(t))⌋)
```
Familiar + healthy → 7–10 steps. Novel + degraded → 2–3 steps. **Temporal ensembling** (overlapping chunks averaged over time) smooths transitions between successive commitments, preventing jerky behavior at chunk boundaries. Computational analog to the ~400ms biological integration window: the tradeoff between deliberation depth and responsiveness, with the tradeoff itself learned.

**Dual-utility PPO.** Separate value heads `V_mission`, `V_self` with independent value loss coefficients to prevent one head from dominating gradient signal:
```
advantage  = α(t)·(R_mission − V_mission) + (1−α(t))·(R_self − V_self)
L_value    = c_m·MSE(V_mission, G_mission) + c_s·MSE(V_self, G_self)
```

**Exchange rate `α(t)`:**

| Condition | `α(t)` | Behavior |
|---|---|---|
| Healthy, early mission, full battery | ~0.85 | Mission dominant |
| Moderate wear, mid-mission | ~0.55 | Balanced |
| Degraded, late mission, low battery | ~0.20 | Self-preservation dominant |
| High `ε_ood` | Lower | Additional epistemic caution |

The shift from bold to conservative is emergent from training, not rule-programmed. Addresses **PS-5**. Combined with `ε_ood` and competence memory distance, `α(t)` also governs the return commitment problem (**PS-2**): the agent's decision of when to reverse is a function of the covariance between map uncertainty, remaining resource estimate, and self-model degradation — all of which flow through `Z_relation` into `α(t)`.

### 4.6 Reality Alignment

After every action chunk:
```
L_reality = dist(predicted_outcome(Z_relation), observed_physical_outcome)
```
Because `Z_relation` was produced by cross-attention over both `Z_world` and `Z_self`, gradients from `L_reality` flow back through the relation module into **both** the world module and the self module simultaneously. The joint update is not a design choice — it is a mathematical consequence of the cross-attention coupling. This is the architectural analog of postdiction (Section 1.3). Over accumulated experience, cross-attention learns to weight the two streams to minimize prediction error. Continual learning uses rank-8 LoRA adapters [17] to prevent catastrophic forgetting.

### 4.7 Growth Loop

```
Perceive → Self-assess → Skill boundary query
    ↓ known              ↓ variation            ↓ novel
    Execute               Adapt                  Improvise → Write to C
                    ↓
              Act → Observe outcome → L_reality
                    ↓
              Update world + self models → (loop)
```

Every pass changes the agent. Every successful improvisation becomes a retrievable skill. Every reality correction refines all models. The agent at hour six carries the accumulated evidence of everything it has experienced — its decisions are the product of that history, not a static deployment-time function.

---

## 5. Implementation

### 5.1 Platform

| Component | Specification |
|---|---|
| Base | TurtleBot4 / custom differential drive |
| LiDAR | Ouster OS1-32 (32-beam, 10 Hz) |
| Stereo | Intel RealSense D435i |
| IMU | Vectornav VN-100 (200 Hz) |
| Ultrasonic | HC-SR04 × 4 |
| Compute | NVIDIA Jetson Orin NX 16 GB |
| Power | 4S LiPo + BMS |
| Framework | ROS2 Humble |

### 5.2 ROS2 Node Architecture

| Node | Rate | Responsibility |
|---|---|---|
| `world_module_node` | 10 Hz | VoxFormer inference → `Z_world` |
| `self_module_node` | 20 Hz | `H(t)`, `Z_self`, `ε_ood` |
| `decision_node` | 5 Hz | ACT+PPO → action chunks |
| `reality_alignment_node` | 1 Hz | Continual gradient updates |
| `skill_memory_node` | On-demand | Competence memory server |
| `embodiment_node` | Platform | Hardware abstraction |
| `mission_monitor_node` | 2 Hz | Resource budget, return trigger |

Rates are nominal. `self_module_node` and `decision_node` increase frequency proportionally to `ε_ood` (Section 4.3).

### 5.3 Compute Budget

| Module | Latency | Freq | GPU Mem |
|---|---|---|---|
| World (TRT fp16) | ~45 ms | 10 Hz | 1.2 GB |
| Self | ~8 ms | 20 Hz | 0.3 GB |
| Decision | ~35 ms | 5 Hz | 0.8 GB |
| Reality Alignment | ~200 ms | 1 Hz | 0.4 GB |
| Competence Memory (FAISS) | ~2 ms | On-demand | 0.2 GB |

### 5.4 Cross-Embodiment Interface

```python
class EmbodimentInterface(ABC):
    def get_sensor_bundle(self) -> SensorBundle: ...
    def get_homeostatic_state(self) -> HomeostaticState: ...   # H(t) ∈ [0,1]^6
    def execute_action(self, chunk: ActionChunk) -> None: ...
    def get_capability_envelope(self) -> CapabilityEnvelope: ...
    def get_physics_constraints(self) -> PhysicsConstraints: ...
```

| Embodiment | Sensors | Action Space |
|---|---|---|
| Wheeled | LiDAR, stereo, IMU, ultrasonic, encoders | `(v_x, ω)` |
| Aerial | Depth camera, barometer, optical flow | `(v_x, v_y, v_z, yaw)` |
| Aquatic | Sonar, pressure, DVL | `(thrust, depth)` |
| Humanoid | RGB-D, force/torque, tactile | Joint positions |

The cognitive architecture is identical across embodiments. Only the embodiment node is swapped.

### 5.5 Communication Loss

Core decision loop continues uninterrupted. `L_reality` gradients accumulate locally; competence memory writes persist to local storage. Accumulated updates apply on reconnection.

---

## 6. Case Study

### 6.1 The Point of No Return

**Scenario.** Hour 4, battery 41%, `σ_kinematic = 0.71`. Encounter: 28° downward slope, loose shale, high-value target beyond.

**Without PerSPARAMA.** Static traversability: 67% passable. Robot descends. Mid-slope slip corrupts localization by 40 cm/2s. At bottom: 28° upward return, battery 31%, elevated motor temperature. No asymmetric return model. Stranded.

**With PerSPARAMA.**
- *Perception:* `q_self` embeds `H(t) = [0.74, 0.68, 0.41, 0.71, 0.88, 0.61]`. Slope voxels carry degraded kinematic provenance.
- *Competence query:* Nearest entries — gravel 22° at `H_t≈[0.91,...]` (success), wet rock 31° at `H_t=[0.55, 0.61, 0.38,...]` (partial failure). Weighted `P(success) = 0.52`.
- *Dual utility:* `α(t) = 0.31` (self-dominant). Return probability drops from 0.81 to 0.54 post-descent.
- *Decision:* Do not descend. Map from current position. Return via known path. Target logged for next mission with better margin.

### 6.2 Mission Growth Trace

| Hour | Event | Update |
|---|---|---|
| 0 | Deploy | `H(t)=[1,1,1,1,1,1]`, `C=∅` |
| 1 | Wet limestone | `σ_kinematic↓`; first novel `C` entry |
| 2 | LiDAR intermittent | `η_sensor↓`; retroactive map uncertainty revision |
| 3 | Narrow passage success | High-value `C` entry; `Δ↓` for similar situations |
| 4 | Slope (above) | Self-dominant decision; target deferred |
| 5 | Return | Uses hour-3 entry; succeeds |
| 6 | Complete | `|C| += 47`; `α(t)` recalibrated |

### 6.3 Testable Predictions

| Prediction | Test | Expected |
|---|---|---|
| `α(t)` monotone with battery | Vary `V_battery` | `α↓` monotonically |
| `ε_ood↑` → shorter chunks | Force novel terrain | `chunk_len ≤ 4` at `ε_ood > 0.8` |
| Second encounter → lower `Δ` | Same terrain twice | `Δ₂ < Δ₁` |
| Degraded kinematics → early return | `σ_kinematic < 0.4` | `V_battery_return > 0.35` |
| Improvisation → skill written | `Δ > 0.7` | `|C|` increases |
| Morality floor active | `α = 0`; env. damage | `R_total < R_self_only` |

---

## 7. Discussion

### 7.1 Self-Worth as Utility, Not Constraint

Constraint-based safety defines prohibited states and prevents entry via hard boundaries. Such systems sacrifice self-preservation whenever the constraint is not violated — they have no mechanism to *value* continuity in proportion to accumulated capability. PerSPARAMA's utility-based formulation (Section 1.2.2) weights self-preservation proportionally to current physical margin via `α(t)`, producing qualitatively different behavior precisely in the degraded regime where safety matters most.

### 7.2 Situated Perception

The self-as-token mechanism (Section 4.2) instantiates Section 1.2.1 at the perceptual level. By making self-conditioning explicit and differentiable, `Z_world` is never "objective" in a way that is dishonest about the observer. This is not a limitation to engineer around — it is the fundamental epistemic condition of situated agency. The architecture is built to revise, not to defend.

### 7.3 Hope, Made Computational

The three governing principles (Section 1.2) converge to a single architectural property: PerSPARAMA is a deterministic system loaded with honest models and genuine values before the physics runs forward — reality alignment (1.2.1), homeostatic self-knowledge (1.2.2), self-worth as utility (1.2.2), developmental growth (1.2.3). That loading changes the probability distribution over future outcomes. That is hope, made computational.

---

## 8. Limitations

| Limitation | Mitigation Direction |
|---|---|
| Sim-to-real gap: `α(t)`, `ε_ood` trained in simulation | Domain randomization; real-world fine-tuning |
| Cross-embodiment claims architectural only | Aerial/aquatic embodiment validation |
| Competence memory cold start | Simulation pre-population |
| Local gradient accumulation drift | Federated gradient compression |
| `σ_kinematic` requires full odometry suite | Modality-dropout training |
| PS-6 retroactive revision formulated, not implemented | Factor graph propagation on edge hardware |

---

## 9. Conclusion

PerSPARAMA couples sparse voxel perception, homeostatic self-modeling, and dual-utility decision-making through bidirectional cross-attention, producing an embodiment-agnostic cognitive architecture in which perception is conditioned on physical state, self-worth modulates risk tolerance via a learned exchange rate, and the agent grows through reality-aligned continual learning. Seven problem statements defining the failure class of solo agents operating without external validation are addressed within a unified closed-loop framework. The case study demonstrates emergent risk calibration: mission-dominant when margins are high, self-preserving when margins are low, and always conditioned on the agent's honest assessment of who it currently is.

**Future work:** cross-embodiment validation (aerial, aquatic), sim-to-real characterization of `α(t)` and `ε_ood`, PS-6 retroactive map revision on edge hardware, and cross-mission fleet-level competence memory persistence.

> *Any agent exists at the intersection of a reality it can only partially perceive, a self it can only approximately know, and a mission that gives its decisions direction. Its intelligence is the quality of the loop that runs between them. Its growth is what happens to that loop over time.*

---

## Acknowledgments

This research is conducted at the **LOGOS Lab, Arizona State University**. The author thanks the robotics community for foundational contributions in VoxFormer, ACT, and ACT-LoRA. Simulation environments: Gazebo (OSRF) and NVIDIA Isaac Sim.

---

## References

[1] Li et al. "VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion." arXiv:2302.12251, 2023.
[2] Gervet et al. "Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation." CoRL, 2023.
[3] Chen et al. "RISE: 3D Perception Makes Real-World Robot Imitation Simple and Effective." ICRA, 2024.
[4] Zhao, Kumar, Levine, Finn. "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware." RSS, 2023. arXiv:2304.13705.
[5] Gu, Kondepudi, Huang, Gopalan. "Continual Skill and Task Learning via Dialogue." CoRL, 2024. arXiv:2409.03166.
[6] Liu et al. "TAIL: Task-Specific Adapters for Imitation Learning with Large Pretrained Models." 2024.
[7] Wan et al. "LOTUS: Continual Imitation Learning via Unsupervised Skill Discovery." ICLR, 2024.
[8] Xu et al. "XSkill: Cross Embodiment Skill Discovery." CoRL, 2023.
[9] Li et al. "A Comprehensive Survey on World Models for Embodied AI." arXiv:2510.16732, 2025.
[10] "Embodied AI: From LLMs to World Models." arXiv:2509.20021, 2025.
[11] Liu et al. "Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment." IJRR, 2025.
[12] Hajizada et al. "Continual Learning for Autonomous Robots: A Prototype-based Approach." arXiv:2404.00418, 2024.
[13] "Task-Driven Detection of Distribution Shifts With Statistical Guarantees for Robot Learning." IEEE TRO, 2024.
[14] Shojaei Miandashti et al. "Uncertainty Estimation and OOD Detection for LiDAR Segmentation." ECCV-W, 2024.
[15] "Guaranteeing OOD Detection in Deep RL via Transition Estimation." arXiv:2503.05238, 2025.
[16] Salehi et al. "A System-Level View on OOD Data in Robotics." 2023.
[17] Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.
[18] Driess et al. "PaLME: An Embodied Multimodal Language Model." ICML, 2023.
[19] Brohan et al. "RT-1: Robotics Transformer for Real-World Control at Scale." 2023.
[20] Brohan et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." 2023.
[21] Hansen et al. "TDMPC2: Scalable, Robust World Models for Continuous Control." ICLR, 2024.
[22] Zhu et al. "Advancing Autonomy Through Lifelong Learning." Frontiers in Neurorobotics, 2024.
[23] "Large Model Empowered Embodied AI: Decision-Making Survey." arXiv:2508.10399, 2025.
[24] "R²AI: Towards Resistant and Resilient AI." 2025.
[25] Vaswani et al. "Attention Is All You Need." NeurIPS, 2017.
[26] Fang et al. "RH20T: A Comprehensive Robotic Dataset." 2023.

---

## Appendix

### A. Abbreviations

| Abbr. | Definition |
|---|---|
| ACT | Action Chunking Transformer |
| CVAE | Conditional Variational Autoencoder |
| FAISS | Facebook AI Similarity Search |
| H(t) | Homeostatic State Vector |
| LoRA | Low-Rank Adaptation |
| OOD | Out-of-Distribution |
| PerSPARAMA | Perception of a Self worth Perspective Agent with Reality Aligned Mission Autonomy |
| PPO | Proximal Policy Optimization |

### B. ROS2 Messages

```
# HomeostaticState.msg
float32 T_thermal        # [0,1]
float32 I_wheel          # [0,1]
float32 V_battery        # [0,1]
float32 sigma_kinematic  # [0,1]
float32 eta_sensor       # [0,1]
float32 tau_stress       # [0,1]
float32 epsilon_ood      # [0,1]
bool    kinematics_suspect

# ActionChunk.msg
int32     chunk_length
float32[] velocity_x
float32[] velocity_y
float32[] omega
float32[] sensor_weights
float32   alpha
float32   V_mission
float32   V_self
```

### C. Repository Structure

```
PerSPARAMA/
├── persparama_core/
│   ├── models/
│   │   ├── world_module.py
│   │   ├── self_module.py
│   │   ├── competence_memory.py
│   │   ├── cross_attention.py
│   │   ├── act_transformer.py
│   │   └── exchange_rate_net.py
│   └── rl/
│       ├── dual_utility_env.py
│       ├── dual_utility_reward.py
│       └── ppo_trainer.py
├── ros2_ws/src/
│   ├── persparama_perception/
│   ├── persparama_self/
│   ├── persparama_decision/
│   ├── persparama_embodiment/
│   └── persparama_msgs/
├── training/
├── tests/
├── sim/
└── docker/
```

---

<div align="center">

*PerSPARAMA — 2026 | Ashish Paka | LOGOS Lab, Arizona State University*

</div>
