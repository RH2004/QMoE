# Hybrid Quantum-Classical Mixture of Experts: Unlocking Topological Advantage via Interference-Based Routing

## Research Abstract
This project investigates the theoretical and empirical advantages of Quantum Mixture of Experts (QMoE) architectures over classical counterparts. Specifically, this research conducts an ablation study using a Hybrid Architecture (Quantum Router + Classical Experts) to isolate the source of quantum advantage.

While existing literature proposes fully quantum architectures, this work demonstrates that the primary topological advantage lies in the Quantum Gating Network. By leveraging quantum feature maps (Angle Embedding) and wave interference, the Quantum Router acts as a high-dimensional kernel method, effectively "untangling" non-convex data distributions (such as the "Two Moons" dataset) that standard linear classical routers fail to separate efficiently.

## Key Findings
1.  **Mechanism Isolation:** By using identical Classical Experts for both models, we proved that the superior performance of the QMoE is driven strictly by the Quantum Router's ability to model non-linear decision boundaries.
2.  **Topological Advantage:** On non-linearly separable data, the Quantum Router achieves a smooth, curved decision boundary using fewer parameters than a Classical Router, which relies on inefficient linear cuts.
3.  **Parameter Efficiency:** The Quantum Router demonstrates a higher Effective Dimension, achieving comparable or superior accuracy with significantly fewer trainable parameters than a classical network.

## Repository Structure

This repository is organized into three phases, each contained in a specific Jupyter Notebook designed to run on Google Colab (Free Tier) using PennyLane with lightning.qubit acceleration and Adjoint Differentiation.

### Phase 1: Baseline Comparison & Feasibility (QMoE.ipynb)
**Objective:** Establish a performance baseline between Classical MoE, Standard QNNs, and QMoE on simple linear datasets (MNIST/Synthetic).
* **Methodology:** Trained models on linearly separable data.
* **Result:** Classical models outperform Quantum models in wall-clock training time and convergence on simple tasks. This eliminates "Quantum Hype" and motivates the need for a problem class where quantum mechanics offers a genuine mathematical advantage (Phase 2).

### Phase 2: The Interference Hypothesis (Notebook_Phase_2_Proving_the_Quantum_Routing_Advantage.ipynb)
**Objective:** Test the hypothesis that Quantum Routing utilizes interference to solve non-convex topologies that break linear classifiers.
* **Dataset:** "Two Moons" (Interlocked non-linear data).
* **Architecture:**
    * Model A (Control): Classical Linear Router + Classical Linear Experts.
    * Model B (Test): Quantum Router (Angle Embedding) + Classical Linear Experts.
* **Result:** Visual proof of Decision Boundary Divergence. The Classical Router fails (draws straight lines), while the Quantum Router successfully wraps the decision boundary around the data curvature, validating the interference hypothesis.

### Phase 3: Scalability & Noise Resilience (Notebook_Phase_3_Robustness_Verification.ipynb)
**Objective:** Stress-test the Hybrid QMoE to ensure scientific validity and NISQ readiness.
* **Experiment A (Efficiency):** A parameter sweep comparing Accuracy vs. Number of Parameters. Demonstrates that QMoE is more parameter-efficient (higher expressivity per weight).
* **Experiment B (Noise):** Introduction of simulated DepolarizingChannel noise to the quantum circuit.
* **Result:** Establishes the "breakdown point" of the architecture, proving it is robust enough for near-term noisy hardware (NISQ) up to specific error rates.

## Technical Implementation

* **Frameworks:** PyTorch, PennyLane (Xanadu).
* **Backend:** lightning.qubit / lightning.gpu (C++ optimized state-vector simulators).
* **Differentiation:** Adjoint Differentiation (diff_method="adjoint") used to reduce gradient calculation complexity from O(P) to O(1) circuit executions, enabling feasible training times on Colab.

### Installation
To replicate these results, install the required dependencies:

## Installation
```bash
pip install pennylane torch matplotlib scikit-learn pennylane-lightning
```

## The Theory: Why This Works

### The Interference Hypothesis

A classical gating network (typically Softmax(Wx + b)) partitions the input space using hyperplanes (straight lines). To separate interlocked data (like spirals or moons), it requires deep layers to approximate the curve.

A Quantum Router maps classical data x into a quantum state |ψ(x)⟩ via a feature map (e.g., Angle Embedding). This projects the data onto the high-dimensional surface of the Bloch sphere. The router then applies unitary rotations U(theta). The probability of selecting an expert is determined by measurement:

$$P(\\text{expert } i) = |\\langle i | U(\\theta) |\\psi(x)\\rangle|^2$$

Because the amplitudes are complex numbers, they can undergo constructive and destructive interference. This allows the router to create "exclusion zones" and non-linear topology in the original feature space with a very shallow circuit, effectively acting as a powerful learnable kernel.

**Citation & Contact:**
- **Author:** Reda HEDDAD
- **Supervisor:** Dr.Lamiae Bouanane
- **Institution:** Al Akhawayn University

Based on structural concepts from Nguyen et al. (2025) but implementing a novel hybrid ablation study for mechanism isolation.
