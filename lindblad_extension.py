import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation


# ─────────────────────────────────────────────────────────────────────────────
#  Lindblad Decoherence Extension
#  Attach to a Quantum_numerical_simulator instance, or use as a standalone
#  subclass.  See LindbladSolver docstring for a minimal usage example.
# ─────────────────────────────────────────────────────────────────────────────


class LindbladSolver:
    """
    Lindblad master-equation solver for open quantum systems.

    Wraps a ``Quantum_numerical_simulator`` instance and adds dissipative
    time evolution via the Lindblad master equation:

        dρ/dt = -i[H, ρ]  +  Σ_k γ_k ( L_k ρ L_k† − ½ {L_k†L_k, ρ} )

    Supported jump operators
    ------------------------
    - **Dephasing**        L = x̂            (pure phase noise / momentum diffusion)
    - **Amplitude damping** L = â            (energy loss, T₁ decay)
    - **Heating**          L = â†            (energy gain from a warm environment)
    - **Custom**           any (L, γ) pair   (pass via ``extra_ops``)

    Quick-start example
    -------------------
    ::

        sim = Quantum_numerical_simulator(L=256, x_min=-8, x_max=8)
        sim.discrete_Potential(omega=1.0)
        sim.build_hamiltonian()
        sim.compute_eigenstates()

        solver = LindbladSolver(sim)
        solver.build_jump_operators(gamma_dephasing=0.05, gamma_damping=0.02)

        psi0 = sim.coherent_state(x0=2.0, p0=0.0)
        rho0 = np.outer(psi0, psi0.conj())          # pure-state density matrix

        times, rhos = solver.evolve(rho0, t_total=6.0, dt=0.01)
        solver.plot_diagnostics(times, rhos)
        solver.plot_wigner_dm(rhos[0],  title='t = 0  (coherent)')
        solver.plot_wigner_dm(rhos[-1], title='t = T  (decohered)')
        solver.animate_wigner_dm(times, rhos, stride=4)
    """

    def __init__(self, sim):
        """
        Parameters
        ----------
        sim : Quantum_numerical_simulator
            Must already have ``H``, ``eigenvalues``, ``eigenvectors``,
            ``x_discretization``, and ``dx`` set (i.e. call
            ``build_hamiltonian()`` and ``compute_eigenstates()`` first).
        """
        self.sim = sim
        self.jump_operators: list[tuple[np.ndarray, float]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Jump operator construction
    # ─────────────────────────────────────────────────────────────────────────

    def build_jump_operators(
        self,
        gamma_dephasing: float = 0.0,
        gamma_damping: float = 0.0,
        gamma_heating: float = 0.0,
        n_fock: int = None,
        extra_ops: list = None,
    ) -> list:
        """
        Assemble the list of ``(L_k, γ_k)`` jump operators.

        Parameters
        ----------
        gamma_dephasing : float
            Rate for L = x̂  (dephasing / momentum diffusion).
        gamma_damping : float
            Rate for L = â  (amplitude damping, T₁ energy decay).
        gamma_heating : float
            Rate for L = â† (heating from a thermal environment).
        n_fock : int, optional
            Number of Fock levels used to build the ladder operators.
            Defaults to ``min(30, L)``.
        extra_ops : list of (ndarray, float), optional
            Additional ``(L_matrix, gamma)`` pairs for custom jump operators.

        Returns
        -------
        list of (ndarray, float)
            The assembled jump operator list (also stored as
            ``self.jump_operators``).
        """
        sim = self.sim
        evec = sim.eigenvectors          # columns = position-basis eigenstates
        n_fock = n_fock or min(30, sim.L)

        self.jump_operators = []

        if gamma_dephasing > 0:
            L_deph = np.diag(sim.x_discretization.astype(complex))
            self.jump_operators.append((L_deph, gamma_dephasing))
            print(f"[Lindblad] Dephasing   L = x̂,  γ = {gamma_dephasing}")

        if gamma_damping > 0:
            L_lower = self._build_ladder(evec, n_fock, raising=False)
            self.jump_operators.append((L_lower, gamma_damping))
            print(f"[Lindblad] Damping     L = â,   γ = {gamma_damping}")

        if gamma_heating > 0:
            L_raise = self._build_ladder(evec, n_fock, raising=True)
            self.jump_operators.append((L_raise, gamma_heating))
            print(f"[Lindblad] Heating     L = â†,  γ = {gamma_heating}")

        if extra_ops:
            for L_c, g_c in extra_ops:
                self.jump_operators.append((np.asarray(L_c, dtype=complex), g_c))
            print(f"[Lindblad] Added {len(extra_ops)} custom jump operator(s).")

        return self.jump_operators

    @staticmethod
    def _build_ladder(evec: np.ndarray, n_fock: int, raising: bool) -> np.ndarray:
        """
        Build â or â† in the position basis from the eigenvector matrix.

        Uses the harmonic-oscillator ladder relations:

            â  = Σ_{n=0}^{N-2}  √(n+1)  |n⟩⟨n+1|
            â† = Σ_{n=0}^{N-2}  √(n+1)  |n+1⟩⟨n|

        Parameters
        ----------
        evec : ndarray, shape (L, ≥n_fock)
            Eigenvector matrix; column ``n`` is the n-th eigenstate.
        n_fock : int
            Number of Fock levels to include.
        raising : bool
            ``True``  → return â†;  ``False`` → return â.

        Notes
        -----
        ``np.linalg.eigh`` returns eigenvectors with arbitrary relative signs
        between consecutive levels.  Before building the ladder operator we
        fix the signs so that ⟨n−1|x̂|n⟩ > 0 for every n, which is required
        for the ladder relations â|n⟩ = √n|n−1⟩ to hold with the correct phase.
        """
        dim = evec.shape[0]

        # Phase-fix: enforce <n-1|x|n> > 0 for all n.
        # Use the grid index as a proxy for x (same sign structure).
        vecs = evec[:, :n_fock].copy()
        mid = dim // 2
        if vecs[mid, 0] < 0:
            vecs[:, 0] *= -1
        x_idx = np.arange(dim, dtype=float)
        for n in range(1, n_fock):
            me = np.dot(vecs[:, n - 1] * x_idx, vecs[:, n])
            if me < 0:
                vecs[:, n] *= -1

        L_op = np.zeros((dim, dim), dtype=complex)
        for n in range(n_fock - 1):
            amplitude = np.sqrt(n + 1)
            ket_n  = vecs[:, n]
            ket_n1 = vecs[:, n + 1]
            if raising:
                L_op += amplitude * np.outer(ket_n1, ket_n)   # â†
            else:
                L_op += amplitude * np.outer(ket_n,  ket_n1)  # â
        return L_op

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Right-hand side of the master equation
    # ─────────────────────────────────────────────────────────────────────────

    def _lindblad_rhs(self, rho: np.ndarray) -> np.ndarray:
        """
        Evaluate dρ/dt = −i[H, ρ] + Σ_k γ_k D[L_k](ρ).

        The Lindblad dissipator is:

            D[L](ρ) = L ρ L† − ½ (L†L ρ + ρ L†L)
        """
        H = self.sim.H

        # Coherent (unitary) part: -i[H, ρ]
        drho = -1j * (H @ rho - rho @ H)

        # Dissipative part: one term per jump operator
        for L_k, gamma_k in self.jump_operators:
            Ldag = L_k.conj().T
            LdagL = Ldag @ L_k
            drho += gamma_k * (
                L_k @ rho @ Ldag
                - 0.5 * (LdagL @ rho + rho @ LdagL)
            )

        return drho

    # ─────────────────────────────────────────────────────────────────────────
    # 3. RK4 time evolution
    # ─────────────────────────────────────────────────────────────────────────

    def evolve(
        self,
        rho0: np.ndarray,
        t_total: float,
        dt: float,
        stride: int = 1,
    ) -> tuple[np.ndarray, list]:
        """
        Integrate the Lindblad equation with 4th-order Runge-Kutta.

        Parameters
        ----------
        rho0 : ndarray, shape (L, L)
            Initial density matrix (complex).
        t_total : float
            Total simulation time.
        dt : float
            RK4 time step.
        stride : int
            Save every ``stride``-th step to reduce memory usage for long runs.

        Returns
        -------
        times : ndarray, shape (n_saved,)
            Saved time points.
        rhos : list of ndarray
            Density matrices at each saved time point.
        """
        n_steps = int(t_total / dt)
        rho = np.asarray(rho0, dtype=complex).copy()
        times = [0.0]
        rhos = [rho.copy()]

        print(f"[Lindblad] Evolving: {n_steps} steps,  dt = {dt},  stride = {stride} …")

        for step in range(1, n_steps + 1):
            k1 = self._lindblad_rhs(rho)
            k2 = self._lindblad_rhs(rho + 0.5 * dt * k1)
            k3 = self._lindblad_rhs(rho + 0.5 * dt * k2)
            k4 = self._lindblad_rhs(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if step % stride == 0:
                times.append(step * dt)
                rhos.append(rho.copy())

        print("[Lindblad] Evolution complete.")
        return np.array(times), rhos

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Observables from the density matrix
    # ─────────────────────────────────────────────────────────────────────────

    def expectation(self, rho: np.ndarray, O: np.ndarray) -> float:
        """
        Compute ⟨O⟩ = Tr(ρ O).

        Returns the real part; any imaginary residual is numerical noise for
        Hermitian observables.
        """
        return float(np.real(np.trace(rho @ O))) * self.sim.dx

    def purity(self, rho: np.ndarray) -> float:
        """
        Compute purity γ = Tr(ρ²) ∈ (1/L, 1].

        A value of 1 indicates a pure state; smaller values indicate mixing.
        """
        return float(np.real(np.trace(rho @ rho))) * self.sim.dx ** 2

    def von_neumann_entropy(self, rho: np.ndarray, eps: float = 1e-12) -> float:
        """
        Compute von Neumann entropy S = −Tr(ρ ln ρ).

        Zero for pure states; positive and growing as the system decoheres.

        Parameters
        ----------
        eps : float
            Eigenvalues below this threshold are treated as zero (avoids
            ``log(0)`` from numerical noise).
        """
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals * self.sim.dx          # scale to proper probabilities
        eigvals = eigvals[eigvals > eps]
        return float(-np.sum(eigvals * np.log(eigvals)))

    def position_variance(self, rho: np.ndarray) -> float:
        """Compute Var(x) = ⟨x²⟩ − ⟨x⟩²."""
        x = self.sim.x_discretization
        mean_x = self.expectation(rho, np.diag(x))
        mean_x2 = self.expectation(rho, np.diag(x ** 2))
        return mean_x2 - mean_x ** 2

    def compute_diagnostics(self, times: np.ndarray, rhos: list) -> dict:
        """
        Compute purity, entropy, ⟨x⟩, and Var(x) at every saved time point.

        Parameters
        ----------
        times : ndarray
            Time points returned by ``evolve()``.
        rhos : list of ndarray
            Density matrices returned by ``evolve()``.

        Returns
        -------
        dict with keys ``'purity'``, ``'entropy'``, ``'mean_x'``, ``'var_x'``.
        """
        x = self.sim.x_discretization
        X = np.diag(x)
        X2 = np.diag(x ** 2)

        purity = np.array([self.purity(r) for r in rhos])
        entropy = np.array([self.von_neumann_entropy(r) for r in rhos])
        mean_x = np.array([self.expectation(r, X) for r in rhos])
        var_x = np.array([self.expectation(r, X2) for r in rhos]) - mean_x ** 2

        return dict(purity=purity, entropy=entropy, mean_x=mean_x, var_x=var_x)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Wigner function from a density matrix
    # ─────────────────────────────────────────────────────────────────────────

    def compute_wigner_dm(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute the Wigner quasi-probability distribution from a density matrix.

        Uses the formula:

            W(x_i, p_j) = (2/π) ∫ dy  ρ(x_i + y, x_i − y) e^{2ipy}

        evaluated via FFT along the off-diagonal (chord) direction.

        Parameters
        ----------
        rho : ndarray, shape (L, L)
            Density matrix in the position basis.

        Returns
        -------
        W : ndarray, shape (L, L)
            Wigner function indexed as ``W[x_index, p_index]``.
        """
        N = self.sim.L
        dx = self.sim.dx

        # Build the chord kernel row-by-row, then FFT to get the momentum axis.
        # Vectorised: for each centre index i, extract the anti-diagonal strip
        # of rho and place it in a zero-padded array before FFT.
        W = np.zeros((N, N), dtype=float)
        half = N // 2
        ks = np.arange(-half, half)           # chord indices

        for i in range(N):
            ip = i + ks                        # x + y  indices
            im = i - ks                        # x - y  indices
            valid = (ip >= 0) & (ip < N) & (im >= 0) & (im < N)
            kernel = np.zeros(N, dtype=complex)
            kernel[ks[valid] + half] = rho[ip[valid], im[valid]]
            W[i, :] = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(kernel))))

        W *= 2 * dx / np.pi
        return W

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Plotting
    # ─────────────────────────────────────────────────────────────────────────

    def _momentum_axis(self) -> np.ndarray:
        """Return the momentum axis p corresponding to the position grid."""
        return np.linspace(-np.pi / (2 * self.sim.dx), np.pi / (2 * self.sim.dx), self.sim.L)

    def plot_wigner_dm(self, rho: np.ndarray, title: str = 'Wigner Function (ρ)'):
        """
        Contour plot of the Wigner function for a given density matrix.

        Parameters
        ----------
        rho : ndarray, shape (L, L)
        title : str
        """
        W = self.compute_wigner_dm(rho)
        x = self.sim.x_discretization
        p = self._momentum_axis()
        lim = np.max(np.abs(W))

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(x, p, W.T, levels=60, cmap='RdBu_r', vmin=-lim, vmax=lim)
        plt.colorbar(cf, ax=ax, label='W(x, p)')
        ax.set_xlabel('x', fontsize=13, fontweight='bold')
        ax.set_ylabel('p', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        plt.tight_layout()
        plt.show()

    def plot_diagnostics(
        self,
        times: np.ndarray,
        rhos: list,
        diag: dict = None,
    ):
        """
        Four-panel diagnostics plot showing decoherence over time.

        Panels:
          1. Purity  γ = Tr(ρ²)
          2. Von Neumann entropy  S
          3. Mean position  ⟨x⟩  (with classical trajectory overlay)
          4. Position variance  Var(x)

        Parameters
        ----------
        times : ndarray
        rhos : list of ndarray
        diag : dict, optional
            Pre-computed output of ``compute_diagnostics()``.  If ``None``,
            it is computed automatically.
        """
        if diag is None:
            diag = self.compute_diagnostics(times, rhos)

        omega = self.sim.omega
        x0_init = float(diag['mean_x'][0])

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

        # ── Purity ──────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, diag['purity'], color='royalblue', lw=2)
        ax1.axhline(1.0, color='gray', ls='--', lw=1, label='Pure state')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Purity  Tr(ρ²)', fontsize=12)
        ax1.set_title('Purity Decay', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ── Von Neumann entropy ──────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, diag['entropy'], color='tomato', lw=2)
        ax2.axhline(0.0, color='gray', ls='--', lw=1, label='Pure state')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Von Neumann Entropy  S', fontsize=12)
        ax2.set_title('Entropy Growth', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ── Mean position ────────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(times, diag['mean_x'], color='seagreen', lw=2)
        ax3.plot(
            times,
            x0_init * np.cos(omega * times),
            color='seagreen', ls='--', lw=1, alpha=0.5,
            label='Classical (undamped)',
        )
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('⟨x⟩', fontsize=12)
        ax3.set_title('Mean Position', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # ── Position variance ────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(times, diag['var_x'], color='darkorchid', lw=2)
        ax4.axhline(
            0.5 / omega, color='gray', ls='--', lw=1,
            label=f'Ground-state  Var = {0.5 / omega:.2f}',
        )
        ax4.set_xlabel('Time', fontsize=12)
        ax4.set_ylabel('Var(x)', fontsize=12)
        ax4.set_title('Position Variance', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        fig.suptitle(
            'Lindblad Decoherence Diagnostics',
            fontsize=15, fontweight='bold', y=1.01,
        )
        plt.tight_layout()
        plt.show()

    def plot_wigner_snapshot_grid(
        self,
        times: np.ndarray,
        rhos: list,
        n_snapshots: int = 6,
    ):
        """
        Grid of Wigner function snapshots at equally spaced times.

        Useful for visualising the gradual loss of phase-space coherence.

        Parameters
        ----------
        times : ndarray
        rhos : list of ndarray
        n_snapshots : int
            Number of panels in the grid (default 6).
        """
        indices = np.linspace(0, len(rhos) - 1, n_snapshots, dtype=int)
        x = self.sim.x_discretization
        p = self._momentum_axis()

        W_all = [self.compute_wigner_dm(rhos[i]) for i in indices]
        lim = max(np.max(np.abs(W)) for W in W_all)

        ncols = 3
        nrows = int(np.ceil(n_snapshots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
        axes = np.asarray(axes).flatten()

        for j, (idx, W) in enumerate(zip(indices, W_all)):
            ax = axes[j]
            ax.contourf(x, p, W.T, levels=50, cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_title(f't = {times[idx]:.2f}', fontsize=11, fontweight='bold')
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('p', fontsize=10)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)

        for ax in axes[n_snapshots:]:
            ax.set_visible(False)

        fig.suptitle(
            'Wigner Function Snapshots (Lindblad Evolution)',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Phase-space animation
    # ─────────────────────────────────────────────────────────────────────────

    def animate_wigner_dm(
        self,
        times: np.ndarray,
        rhos: list,
        stride: int = 1,
        title: str = 'Wigner Function (Lindblad)',
        save_path: str = None,
    ):
        """
        Animate the Wigner function over the saved density-matrix frames.

        Parameters
        ----------
        times : ndarray
        rhos : list of ndarray
        stride : int
            Use every ``stride``-th frame to speed up the animation.
        title : str
        save_path : str, optional
            If provided, saves the animation as a ``.gif`` (requires Pillow).

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        idx = list(range(0, len(rhos), stride))
        rhos_sub = [rhos[i] for i in idx]
        times_sub = times[idx]

        x = self.sim.x_discretization
        p = self._momentum_axis()

        print("[Lindblad] Pre-computing Wigner frames …")
        W_frames = [self.compute_wigner_dm(r) for r in rhos_sub]
        lim = max(np.max(np.abs(W)) for W in W_frames)
        print("[Lindblad] Starting animation …")

        fig, ax = plt.subplots(figsize=(7, 6))

        def update(frame):
            ax.clear()
            ax.contourf(x, p, W_frames[frame].T,
                        levels=50, cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_xlabel('x', fontsize=12, fontweight='bold')
            ax.set_ylabel('p', fontsize=12, fontweight='bold')
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_title(
                f'{title} — t = {times_sub[frame]:.3f}',
                fontsize=13, fontweight='bold',
            )

        anim = FuncAnimation(fig, update, frames=len(W_frames), interval=60)
        plt.tight_layout()

        if save_path is not None:
            print(f"[Lindblad] Saving animation to '{save_path}' …")
            anim.save(save_path, writer='pillow', fps=15)
            print("[Lindblad] Saved.")

        plt.show()
        return anim


# ─────────────────────────────────────────────────────────────────────────────
# Quick-start demo  (run with:  python lindblad_extension.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    try:
        from QuantumSimulator import Quantum_numerical_simulator
    except ImportError:
        raise SystemExit(
            "QuantumSimulator.py not found.  "
            "Place it in the same directory and re-run."
        )

    # 1. Build simulator
    sim = Quantum_numerical_simulator(L=200, x_min=-8, x_max=8)
    sim.discrete_Potential(omega=1.0)
    sim.build_hamiltonian()
    sim.compute_eigenstates()

    # 2. Initial coherent state as a pure density matrix
    psi0 = sim.coherent_state(x0=2.0, p0=0.0)
    rho0 = np.outer(psi0, psi0.conj())

    # 3. Attach solver and define jump operators
    solver = LindbladSolver(sim)
    solver.build_jump_operators(
        gamma_dephasing=0.05,   # phase noise
        gamma_damping=0.02,     # T₁ energy decay
    )

    # 4. Evolve for three classical periods
    T_cl = 2 * np.pi / sim.omega
    times, rhos = solver.evolve(rho0, t_total=3 * T_cl, dt=0.005, stride=5)

    # 5. Diagnostics
    solver.plot_diagnostics(times, rhos)

    # 6. Wigner snapshots
    solver.plot_wigner_snapshot_grid(times, rhos, n_snapshots=6)

    # 7. Wigner at start and end
    solver.plot_wigner_dm(rhos[0],  title='Wigner — t = 0  (pure coherent state)')
    solver.plot_wigner_dm(rhos[-1], title=f'Wigner — t = {times[-1]:.2f}  (decohered)')

    # 8. Animated phase-space evolution (uncomment to run)
    # solver.animate_wigner_dm(times, rhos, stride=2, save_path='decoherence.gif')
