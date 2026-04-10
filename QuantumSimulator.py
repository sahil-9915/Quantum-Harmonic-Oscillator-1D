import numpy as np
import matplotlib.pyplot as plt

class Quantum_numerical_simulator:
     
     
     def __init__(self, L, x_min, x_max):
         
         self.L=L
         self.x_min = x_min
         self.x_max = x_max
         
         self.approx_time_step=None
         self.approx_evolution=None
         self.pdf= None   #probability density
         self.sigma=None
         self.x0=None
         self.omega=None
         self.tau=None
         self.tau_default=0.00025
         self.timesteps=None
         self.x_discretization = None
         self.dx = None
         self.V = None
         self.init_wavefunction = None
         self.evolved_wavefunction = None
         self.norm = None
         self.expectation_x = None
         self.expectation_x2 = None
         self.time_discretization = np.linspace(0, 10, 101)

         
         self.discretization()
        

     def discretization(self):

        self.dx = (self.x_max - self.x_min) / (self.L - 1)
        self.x_discretization = np.linspace(self.x_min, self.x_max, self.L)
        


     def Initial_gaussian_wavefunction(self, sigma, x0):
        self.init_wavefunction=np.zeros(len(self.x_discretization))
        self.norm=0
        n=0
        for i in self.x_discretization:
            self.init_wavefunction[n]= (np.pi * sigma**2)**(-0.25) * np.exp(-(i - x0)**2 / (2 * sigma**2))
            n=n+1
        self.evolved_wavefunction=self.init_wavefunction


     def compute_norm(self):
        psi_squared = np.abs(self.evolved_wavefunction)**2
        norm_squared = np.trapezoid(psi_squared, self.x_discretization)
        self.norm = np.sqrt(norm_squared)
        return self.norm
     
     
     def plot_probability_density(self):
         prob_density = self.pdf
         fig, ax = plt.subplots(figsize=(12, 7))
         ax.plot(self.x_discretization, prob_density, color='darkblue', linewidth=2.5, label='|ψ(x)|²')
         ax.fill_between (self.x_discretization, 0, prob_density, alpha=0.3, color='skyblue', label='Probability')
         
         ax.set_xlim(self.x_min, self.x_max)
         ax.set_ylim(0, 1.5)

         ax.set_xlabel('Position (x)', fontsize=14, fontweight='bold')
         ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
         ax.grid(True, alpha=0.3, linestyle='--')
         ax.legend(fontsize=12, loc='best')
         plt.show()

     def discrete_Potential(self, omega):      #define the discretized potential
        self.V=  0.5 * (omega**2) * (self.x_discretization**2 ) #define the discretized Harmonic Oscillator potential
        self.omega=omega


     def build_hamiltonian(self):
        diag_main = 1 / self.dx**2 + self.V
        diag_off  = -1 / (2 * self.dx**2) * np.ones(self.L - 1)
        self.H = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
        return self.H

     def compute_eigenstates(self, verbose=False):
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
        if verbose:
            print(f"Computed {len(self.eigenvalues)} eigenstates")
        return self.eigenvalues, self.eigenvectors


     def approximate_evolution(self,tau,time):
        #For time step operator U(t) = exp(-iτH)
        # e^{-iτH} ≈ e^{-iτK₁/2} e^{-iτK₂/2} e^{-iτV} e^{-iτK₂/2} e^{-iτK₁/2}
        # where H = K₁ + K₂ + V is the Hamiltonian and K1, K2 and V are hermitian matrices to ensure approx time step operator is unitary.
        # defining e^{-iτV}
        self.tau=tau
        V_diag = (1 / self.dx**2) * (1 + self.dx**2 * self.V)
        exp_V_diag_1D= np.exp(-1j * tau * V_diag)
        exp_V_diag=np.diag(exp_V_diag_1D)
        # defining e^{-iτK₂/2} and e^{-iτK₁/2}
        a = tau / (4 * self.dx**2)
        c = np.cos(a)
        s = np.sin(a)
        is_val = 1j * s
        exp_K1_half = np.zeros((self.L, self.L), dtype=complex)
        exp_K2_half = np.zeros((self.L, self.L), dtype=complex)
        for i in range(0, self.L-1, 2):
            # Only fill if we have a complete 2×2 block
            if i+1 < self.L:
                exp_K1_half[i, i] = c
                exp_K1_half[i, i+1] = is_val
                exp_K1_half[i+1, i] = is_val
                exp_K1_half[i+1, i+1] = c
        
        # If L is odd, last element is just c
        if self.L % 2 == 1:
            exp_K1_half[self.L-1, self.L-1] = 1

        exp_K2_half[0, 0] = 1
        
        for i in range(1, self.L-1, 2):
            # Only fill if we have a complete 2×2 block
            if i+1 < self.L:
                exp_K2_half[i, i] = c
                exp_K2_half[i, i+1] = is_val
                exp_K2_half[i+1, i] = is_val
                exp_K2_half[i+1, i+1] = c
        
        # If L is even, need to handle last element
        if self.L % 2 == 0 and self.L > 0:
            exp_K2_half[self.L-1, self.L-1] = c

        approx_time_step= exp_K1_half @ exp_K2_half @ exp_V_diag @ exp_K2_half @ exp_K1_half
        self.approx_time_step=approx_time_step
        self.approx_evolution = np.eye(self.approx_time_step.shape[0])
        
        self.timesteps= int(time/tau)
        #self.approx_evolution = np.linalg.matrix_power(self.approx_time_step, self.timesteps)    
        #self.evolved_wavefunction = self.approx_evolution @ self.init_wavefunction
        psi = self.init_wavefunction.copy()
        for _ in range(self.timesteps):
            psi = self.approx_time_step @ psi
        self.evolved_wavefunction = psi
        self.pdf = np.abs(self.evolved_wavefunction)**2
        self.expectation_x= np.trapezoid(self.pdf*self.x_discretization, self.x_discretization)
        self.expectation_x2= np.trapezoid(self.pdf*(self.x_discretization**2), self.x_discretization)

        return self.evolved_wavefunction

     def build_time_step(self, tau):
        """
        Build the Trotter time step operator without evolving any wavefunction.
        Use this before calling animate_wigner.

        Parameters
        ----------
        tau : float  time step size
        """
        self.tau = tau
        V_diag = (1 / self.dx**2) * (1 + self.dx**2 * self.V)
        exp_V_diag = np.diag(np.exp(-1j * tau * V_diag))

        a = tau / (4 * self.dx**2)
        c = np.cos(a)
        s = np.sin(a)
        is_val = 1j * s

        exp_K1_half = np.zeros((self.L, self.L), dtype=complex)
        exp_K2_half = np.zeros((self.L, self.L), dtype=complex)

        for i in range(0, self.L-1, 2):
            if i+1 < self.L:
                exp_K1_half[i, i] = c
                exp_K1_half[i, i+1] = is_val
                exp_K1_half[i+1, i] = is_val
                exp_K1_half[i+1, i+1] = c
        if self.L % 2 == 1:
            exp_K1_half[self.L-1, self.L-1] = 1

        exp_K2_half[0, 0] = 1
        for i in range(1, self.L-1, 2):
            if i+1 < self.L:
                exp_K2_half[i, i] = c
                exp_K2_half[i, i+1] = is_val
                exp_K2_half[i+1, i] = is_val
                exp_K2_half[i+1, i+1] = c
        if self.L % 2 == 0 and self.L > 0:
            exp_K2_half[self.L-1, self.L-1] = c

        self.approx_time_step = exp_K1_half @ exp_K2_half @ exp_V_diag @ exp_K2_half @ exp_K1_half


     def plot_expectation_x(self):
         temp_expectation_x_array = np.zeros(self.time_discretization.shape)
         t_wavefunction=self.init_wavefunction
         n=0
         delta= float(self.time_discretization[1]-self.time_discretization[0])
         for t in range(len(self.time_discretization)):
             temp_wavefunction=self.approximate_evolution(self.tau_default,delta)
             self.init_wavefunction=temp_wavefunction
             temp_expectation_x_array[n]=self.expectation_x
             n=n+1

         self.init_wavefunction= t_wavefunction
         
         plt.figure(figsize=(12, 6))
         plt.plot(self.time_discretization, temp_expectation_x_array, 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.7)

         plt.xlabel('Time', fontsize=14, fontweight='bold')
         plt.ylabel('Expectation Value ⟨x⟩', fontsize=14, fontweight='bold')
         plt.title('Time Evolution of Position Expectation Value', fontsize=16, fontweight='bold')

         plt.grid(True, alpha=0.3, linestyle='--')
         plt.xlim(0, 10)
         plt.ylim(-self.omega-0.5,self.omega+0.5)
         plt.show()

     def plot_expectation_var(self):
         temp_expectation_var_array = np.zeros(self.time_discretization.shape)
         t_wavefunction=self.init_wavefunction
         n=0
         delta= float(self.time_discretization[1]-self.time_discretization[0])
         for t in range(len(self.time_discretization)):
             temp_wavefunction=self.approximate_evolution(self.tau_default,delta)
             self.init_wavefunction=temp_wavefunction
             temp_expectation_var_array[n]=self.expectation_x2 - (self.expectation_x**2)
             n=n+1

         self.init_wavefunction= t_wavefunction
         
         plt.figure(figsize=(12, 6))
         plt.plot(self.time_discretization, temp_expectation_var_array, 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.7)

         plt.xlabel('Time', fontsize=14, fontweight='bold')
         plt.ylabel('Expectation Value of Variance', fontsize=14, fontweight='bold')
         plt.title('Time Evolution of Variance', fontsize=16, fontweight='bold')

         plt.grid(True, alpha=0.3, linestyle='--')
         plt.xlim(0, 10)
         plt.ylim(-self.omega-0.5,self.omega+0.5)
         plt.show()

     def plot_eigenstates(self, n_states=5):
         fig, ax = plt.subplots(figsize=(10, 6))
         colors = ['royalblue', 'tomato', 'seagreen', 'darkorchid', 'darkorange']
    
         for n in range(n_states):
            psi_n = self.eigenvectors[:, n]
            if psi_n[self.L//2] < 0:
                psi_n = -psi_n
            ax.plot(self.x_discretization, psi_n, 
                color=colors[n], linewidth=2, label=f'n={n}, E={self.eigenvalues[n]:.3f}')
    
         ax.set_xlabel('x', fontsize=13, fontweight='bold')
         ax.set_ylabel('ψ(x)', fontsize=13, fontweight='bold')
         ax.set_title('Harmonic Oscillator Eigenstates', fontsize=15, fontweight='bold')
         ax.set_xlim(-6, 6)
         ax.set_ylim(-0.20, 0.20)
         ax.legend(fontsize=11, loc='upper right')
         ax.grid(True, alpha=0.3, linestyle='--')
         plt.tight_layout()
         plt.show()


     def compute_wigner(self, psi):
         N = len(psi)
         W = np.zeros((N, N), dtype=float)
    
         for i in range(N):
            kernel = np.zeros(N, dtype=complex)
            for k in range(-(N//2), N//2):
                ip = i + k   # index for X+y
                im = i - k   # index for X-y
                if 0 <= ip < N and 0 <= im < N:
                    kernel[k + N//2] = psi[ip] * np.conj(psi[im])
            W[i, :] = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(kernel))))
    
         W *= 2 * self.dx / np.pi
         return W

     def plot_wigner(self, psi, title='Wigner Function'):
         W = self.compute_wigner(psi)
    
         p_max = np.pi / self.dx
         p = np.linspace(-p_max/2, p_max/2, len(psi))
    
         fig, ax = plt.subplots(figsize=(7, 6))
         limit = np.max(np.abs(W))
         im = ax.contourf(self.x_discretization, p, W.T,
                     levels=50, cmap='RdBu_r',
                     vmin=-limit, vmax=limit)
         plt.colorbar(im, ax=ax, label='W(x,p)')
         ax.set_xlabel('x', fontsize=13, fontweight='bold')
         ax.set_ylabel('p', fontsize=13, fontweight='bold')
         ax.set_title(title, fontsize=14, fontweight='bold')
         ax.set_xlim(-6, 6)
         ax.set_ylim(-6, 6)
         plt.tight_layout()
         plt.show()

     def coherent_state(self, x0, p0=0):
        #Construct a coherent state centered at x0 with momentum p0.

         sigma = 1 / np.sqrt(self.omega)
         psi = ((self.omega / np.pi) ** 0.25 * np.exp(-self.omega * (self.x_discretization - x0)**2 / 2) * np.exp(1j * p0 * self.x_discretization))
         # Normalize
         norm = np.sqrt(np.trapezoid(np.abs(psi)**2, self.x_discretization))
         psi /= norm
         return psi

     def animate_wigner(self, psi0, n_frames=60, title='Wigner Function Evolution', save_path=None):
        """
        Animate the time evolution of the Wigner function in phase space.

        Parameters
        ----------
        psi0      : np.ndarray  initial wavefunction (complex)
        n_frames  : int         number of animation frames (default 60)
        title     : str         plot title
        save_path : str or None if provided, saves animation as .gif

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        from matplotlib.animation import FuncAnimation

        # One full classical period T = 2π/ω
        T = 2 * np.pi / self.omega
        times = np.linspace(0, T, n_frames)

        # approx_time_step must already be built via build_time_step()
        if self.approx_time_step is None:
            raise RuntimeError("Call build_time_step(tau) first to build the time step operator.")

        T = 2 * np.pi / self.omega
        steps_per_frame = max(1, int(T / (n_frames * self.tau)))
        print(f"Precomputing frames (tau={self.tau:.4f}, steps_per_frame={steps_per_frame})...")
        psi = psi0.copy().astype(complex)
        wavefunctions = [psi.copy()]
        for _ in range(n_frames - 1):
            for _ in range(steps_per_frame):
                psi = self.approx_time_step @ psi
            wavefunctions.append(psi.copy())

        print("Computing Wigner functions...")
        wigner_frames = [self.compute_wigner(psi) for psi in wavefunctions]

        p_max = np.pi / self.dx
        p = np.linspace(-p_max / 2, p_max / 2, self.L)
        limit = max(np.max(np.abs(W)) for W in wigner_frames)

        fig, ax = plt.subplots(figsize=(7, 6))

        def update(frame):
            ax.clear()
            ax.contourf(self.x_discretization, p, wigner_frames[frame].T,
                        levels=50, cmap='RdBu_r',
                        vmin=-limit, vmax=limit)
            ax.set_xlabel('x', fontsize=13, fontweight='bold')
            ax.set_ylabel('p', fontsize=13, fontweight='bold')
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_title(f'{title} — t={times[frame]:.2f}', fontsize=14, fontweight='bold')

        anim = FuncAnimation(fig, update, frames=n_frames, interval=50)
        plt.tight_layout()

        if save_path is not None:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
            print("Saved.")

        plt.show()
        return anim

             
         
         
        
     
        
     

 
             
    



    


               
    




    

        






