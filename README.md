# Quantum Harmonic Oscillator — 1D Simulation

A numerical framework for simulating the **1D quantum harmonic oscillator**, covering closed-system unitary evolution, phase-space visualisation via the Wigner function, and open-system decoherence via the Lindblad master equation.

---

## Repository Structure

| File | Description |
|---|---|
| `QuantumSimulator.py` | Core simulator: grid setup, Hamiltonian, unitary evolution, Wigner function |
| `lindblad_extension.py` | Open-system extension: Lindblad master equation, density matrix evolution |
| `simulations.ipynb` | Wavefunction evolution, eigenstates, expectation values |
| `Simulations_in_phase_space.ipynb` | Wigner function evolution for coherent states, eigenstates, and superpositions |
| `Decoherence_simulation.ipynb` | Decoherence simulation via the Lindblad master equation |
| `coherent_evolution.gif` | Animated Wigner function of a coherent state orbiting in phase space |
| `eigenstate_n1_evolution.gif` | Animated Wigner function of the n=1 energy eigenstate |
| `superposition_evolution.gif` | Animated Wigner function of the superposition $(|0\rangle + |1\rangle)/\sqrt{2}$ |

---

## Physics

### Closed system — TDSE

The wavefunction evolves under the time-dependent Schrödinger equation:

$$i\hbar \frac{\partial \psi}{\partial t} = \hat{H}\psi, \qquad \hat{H} = -\frac{1}{2}\frac{d^2}{dx^2} + \frac{1}{2}\omega^2 x^2$$

Time evolution is implemented via a **second-order Trotter splitting** of the unitary operator:

$$e^{-i\tau H} \approx e^{-i\tau K_1/2}\, e^{-i\tau K_2/2}\, e^{-i\tau V}\, e^{-i\tau K_2/2}\, e^{-i\tau K_1/2}$$

where $K_1$, $K_2$, and $V$ are Hermitian sub-operators ensuring the approximation remains unitary. The full evolution over time $T$ is obtained by applying the time-step operator $N = T/\tau$ times.

### Phase space — Wigner function

The Wigner quasi-probability distribution provides a phase-space portrait of the quantum state:

$$W(x, p) = \frac{1}{\pi} \int_{-\infty}^{\infty} \psi^*(x+y)\,\psi(x-y)\, e^{2ipy}\, dy$$

- **Coherent state** — sharp Gaussian blob orbiting the origin; purely classical-like
- **Energy eigenstate n=1** — ring with a negative (blue) core; a signature of non-classical behaviour
- **Superposition $(|0\rangle + |1\rangle)/\sqrt{2}$** — asymmetric blob with an interference fringe; the negative region encodes quantum coherence between the two energy levels

### Open system — Lindblad master equation

When the oscillator is coupled to an environment, the density matrix $\rho$ evolves under:

$$\frac{d\rho}{dt} = -i[\hat{H}, \rho] + \sum_k \gamma_k \mathcal{D}[L_k](\rho)$$

$$\mathcal{D}[L](\rho) = L\rho L^\dagger - \frac{1}{2}\left(L^\dagger L\,\rho + \rho\, L^\dagger L\right)$$

Two noise channels are supported:

| Channel | Jump operator | Physical meaning |
|---|---|---|
| Dephasing | $L = \hat{x}$ | Environment monitors position, destroying phase coherence |
| Amplitude damping | $L = \hat{a}$ | Energy loss into the environment (e.g. photon loss from a cavity) |
| Heating | $L = \hat{a}^\dagger$ | Energy absorption from a warm environment |

Integration uses **4th-order Runge-Kutta**. Diagnostics tracked: purity $\mathrm{Tr}(\rho^2)$, von Neumann entropy $S = -\mathrm{Tr}(\rho \ln \rho)$, mean position $\langle x \rangle$, and position variance $\mathrm{Var}(x)$.

---

## Quick Start

```python
from QuantumSimulator import Quantum_numerical_simulator
from lindblad_extension import LindbladSolver
import numpy as np

# Set up the oscillator
sim = Quantum_numerical_simulator(L=200, x_min=-8, x_max=8)
sim.discrete_Potential(omega=1.0)
sim.build_hamiltonian()
sim.compute_eigenstates()

# --- Closed system: unitary evolution ---
psi0 = sim.coherent_state(x0=2.0, p0=0.0)
sim.init_wavefunction = psi0
sim.build_time_step(tau=0.01)
sim.animate_wigner(psi0, n_frames=60, save_path='coherent_evolution.gif')

# --- Open system: Lindblad decoherence ---
rho0 = np.outer(psi0, psi0.conj())
solver = LindbladSolver(sim)
solver.build_jump_operators(gamma_dephasing=0.05, gamma_damping=0.02, n_fock=80)

T_cl = 2 * np.pi / sim.omega
times, rhos = solver.evolve(rho0, t_total=3*T_cl, dt=0.005, stride=5)
solver.plot_diagnostics(times, rhos)
solver.plot_wigner_dm(rhos[-1], title='Wigner — final decohered state')
```

---
## Requirements

```
numpy
matplotlib
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sahil Sarbadhikary**
- GitHub: [@sahil-9915](https://github.com/sahil-9915)
