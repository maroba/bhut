#!/usr/bin/env python3
"""
Galaxy Simulation using bhut Barnes-Hut N-body accelerator

This script simulates a galaxy of stars using the bhut package for efficient 
gravitational force calculations combined with scipy's adaptive ODE integrators.

Features:
- Adaptive step size integration using scipy.integrate.solve_ivp
- Multiple integration methods (RK45, DOP853, Radau, BDF, LSODA)
- 2D and 3D animation generation
- Energy conservation analysis
- Escaper removal system
- Virial equilibrium initial conditions

Requires: numpy, matplotlib, scipy, bhut
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import bhut


def generate_galaxy_initial_conditions(n_stars=100, galaxy_radius=10.0, mass_per_star=1.0):
    """
    Generate initial conditions for a galaxy simulation.
    
    Creates stars distributed in a disk-like structure with proper
    equilibrium velocities to prevent the galaxy from flying apart.
    
    Parameters
    ----------
    n_stars : int
        Number of stars in the galaxy
    galaxy_radius : float
        Characteristic radius of the galaxy
    mass_per_star : float
        Mass of each star (uniform)
        
    Returns
    -------
    positions : ndarray, shape (n_stars, 3)
        Initial positions of stars
    velocities : ndarray, shape (n_stars, 3) 
        Initial velocities of stars
    masses : ndarray, shape (n_stars,)
        Masses of stars (all equal)
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate masses (all equal)
    masses = np.full(n_stars, mass_per_star, dtype=np.float64)
    total_mass = n_stars * mass_per_star
    
    # Generate positions in a more realistic galaxy disk
    # Use exponential disk profile: surface density ∝ exp(-r/r_scale)
    r_scale = galaxy_radius / 3.0
    
    # Generate radii from exponential distribution
    # Using inverse transform sampling for exponential disk
    u = np.random.uniform(0, 1, n_stars)
    radii = -r_scale * np.log(1 - u * (1 - np.exp(-3)))  # Truncated at 3*r_scale
    
    # Generate angles uniformly
    angles = np.random.uniform(0, 2 * np.pi, n_stars)
    
    # Convert to Cartesian coordinates with thin disk
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.random.normal(0, galaxy_radius * 0.05, n_stars)  # Very thin disk
    
    positions = np.column_stack([x, y, z]).astype(np.float64)
    
    # Generate proper equilibrium velocities
    velocities = np.zeros((n_stars, 3), dtype=np.float64)
    G = 1.0  # Gravitational constant
    
    print("Computing equilibrium velocities...")
    
    for i in range(n_stars):
        r = radii[i]
        if r > 0:
            # Calculate enclosed mass within radius r (for disk profile)
            # For exponential disk: M(r) = M_total * (1 - exp(-r/r_scale) * (1 + r/r_scale))
            enclosed_mass_fraction = 1 - np.exp(-r/r_scale) * (1 + r/r_scale)
            enclosed_mass = total_mass * enclosed_mass_fraction
            
            # Circular velocity from enclosed mass
            if enclosed_mass > 0:
                v_circular = np.sqrt(G * enclosed_mass / r)
            else:
                v_circular = 0
            
            # Add random velocity dispersion (but much smaller)
            v_dispersion = 0.3 * v_circular  # Reduced dispersion
            v_r = np.random.normal(0, v_dispersion * 0.3)  # Radial component
            v_tangential = v_circular + np.random.normal(0, v_dispersion * 0.2)  # Tangential
            v_z = np.random.normal(0, v_dispersion * 0.1)  # Vertical
            
            # Convert to Cartesian coordinates
            cos_angle = np.cos(angles[i])
            sin_angle = np.sin(angles[i])
            
            # Velocity components
            velocities[i, 0] = v_r * cos_angle - v_tangential * sin_angle
            velocities[i, 1] = v_r * sin_angle + v_tangential * cos_angle  
            velocities[i, 2] = v_z
    
    # Apply virial theorem correction for better equilibrium
    # For a gravitationally bound system: 2*KE + PE = 0 (virial theorem)
    print("Applying virial equilibrium correction...")
    
    # Calculate initial potential energy approximately
    potential_energy = 0.0
    for i in range(min(n_stars, 50)):  # Sample for speed
        for j in range(i + 1, min(n_stars, 50)):
            r_vec = positions[i] - positions[j]
            r = np.sqrt(np.sum(r_vec**2) + 0.1**2)  # With softening
            potential_energy -= G * masses[i] * masses[j] / r
    
    # Scale to full system
    potential_energy *= (n_stars * (n_stars - 1)) / (50 * 49) if n_stars > 50 else 1
    
    # Calculate current kinetic energy
    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * np.sum(velocities**2, axis=1)[:, np.newaxis])
    
    # Virial equilibrium: 2*KE + PE = 0 → KE = -PE/2
    target_kinetic = -potential_energy / 2
    
    if kinetic_energy > 0 and target_kinetic > 0:
        velocity_scaling = np.sqrt(target_kinetic / kinetic_energy)
        velocities *= velocity_scaling
        print(f"Applied velocity scaling factor: {velocity_scaling:.3f}")
    
    # Remove center of mass motion
    com_velocity = np.average(velocities, axis=0, weights=masses)
    velocities -= com_velocity
    
    print(f"Generated {n_stars} stars in equilibrium")
    print(f"Galaxy radius scale: {r_scale:.2f}")
    print(f"Velocity dispersion applied: {np.std(velocities):.3f}")
    
    return positions, velocities, masses


def check_virial_equilibrium(positions, velocities, masses, G=1.0, softening=0.1):
    """
    Check if the system satisfies the virial theorem for stability.
    
    For a gravitationally bound system: 2*KE + PE ≈ 0
    Virial ratio Q = -KE/PE should be ≈ 0.5 for equilibrium
    """
    n_stars = len(masses)
    
    # Calculate kinetic energy
    kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    
    # Calculate potential energy
    potential_energy = 0.0
    for i in range(n_stars):
        for j in range(i + 1, n_stars):
            r_vec = positions[i] - positions[j]
            r = np.sqrt(np.sum(r_vec**2) + softening**2)
            potential_energy -= G * masses[i] * masses[j] / r
    
    # Virial ratio
    virial_ratio = -kinetic_energy / potential_energy if potential_energy != 0 else float('inf')
    virial_parameter = 2 * kinetic_energy + potential_energy
    
    print(f"\nVirial Equilibrium Check:")
    print(f"  Kinetic Energy: {kinetic_energy:.4f}")
    print(f"  Potential Energy: {potential_energy:.4f}")
    print(f"  Virial Ratio (KE/|PE|): {virial_ratio:.4f} (should be ~0.5)")
    print(f"  Virial Parameter (2KE+PE): {virial_parameter:.4f} (should be ~0)")
    
    if abs(virial_ratio - 0.5) < 0.2:
        print("  ✓ System appears to be in virial equilibrium")
        return True
    else:
        print("  ⚠ System may not be in equilibrium - could fly apart or collapse")
        return False


def simulate_galaxy(n_stars=100, t_end=1.0, theta=0.5, softening=0.1, galaxy_radius=10.0, 
                   remove_escapers=False, escaper_threshold=2.5, method='RK45', rtol=1e-3, atol=1e-6,
                   progress_steps=10, frames_per_time_unit=50):
    """
    Simulate a galaxy using scipy's ODE integrators with progress reporting.
    
    This version uses chunked integration with scipy.integrate.solve_ivp for progress updates.
    
    Parameters
    ----------
    n_stars : int
        Number of stars in the galaxy
    t_end : float
        Final simulation time
    theta : float
        Barnes-Hut approximation parameter
    softening : float
        Gravitational softening parameter
    galaxy_radius : float
        Characteristic radius of the galaxy (used for escaper removal)
    remove_escapers : bool
        If True, remove stars that escape beyond escaper_threshold * galaxy_radius
    escaper_threshold : float
        Multiplier for galaxy_radius to define escaper removal distance
    method : str
        Integration method ('RK45', 'DOP853', 'Radau', 'BDF', 'LSODA')
    rtol, atol : float
        Relative and absolute tolerance for the integrator
    progress_steps : int
        Number of integration chunks for progress reporting
    frames_per_time_unit : int
        Number of animation frames per simulation time unit (affects animation smoothness)
        
    Returns
    -------
    trajectory : ndarray, shape (n_eval, n_stars, 3)
        Positions at evaluation times
    velocities_history : ndarray, shape (n_eval, n_stars, 3)
        Velocities at evaluation times
    masses : ndarray, shape (n_stars,)
        Final masses
    escaper_counts : list
        Number of remaining stars at each evaluation time
    times : ndarray
        Actual evaluation times used
    """
    print(f"Initializing galaxy with {n_stars} stars...")
    
    # Generate initial conditions
    positions, velocities, masses = generate_galaxy_initial_conditions(n_stars, galaxy_radius)
    
    # Check virial equilibrium
    is_stable = check_virial_equilibrium(positions, velocities, masses, G=1.0, softening=softening)
    
    # Track which stars are still active (not escaped)
    active_mask = np.ones(n_stars, dtype=bool)
    
    def gravitational_ode(t, y):
        """
        ODE function for scipy integrator.
        y = [x1, y1, z1, x2, y2, z2, ..., vx1, vy1, vz1, vx2, vy2, vz2, ...]
        """
        n_active = np.sum(active_mask)
        
        # Extract positions and velocities from the state vector
        pos = y[:3*n_active].reshape(n_active, 3)
        vel = y[3*n_active:].reshape(n_active, 3)
        
        # Handle escaper removal
        if remove_escapers:
            radii = np.linalg.norm(pos, axis=1)
            keep_local = radii < escaper_threshold * galaxy_radius
            
            if not np.all(keep_local):
                # Update the global active mask
                current_active_indices = np.where(active_mask)[0]
                escaped_local_indices = np.where(~keep_local)[0]
                escaped_global_indices = current_active_indices[escaped_local_indices]
                active_mask[escaped_global_indices] = False
                
                # Keep only non-escaped stars
                pos = pos[keep_local]
                vel = vel[keep_local]
                n_active = len(pos)
        
        if n_active == 0:
            return np.zeros_like(y)
        
        # Calculate accelerations using Barnes-Hut
        active_masses = masses[active_mask]
        tree = bhut.Tree(pos, active_masses, leaf_size=16, dim=3)
        tree.build()
        
        accelerations = tree.accelerations(
            theta=theta,
            softening=softening,
            G=1.0
        )
        
        # Construct derivative: [velocities, accelerations]
        dydt = np.zeros(6*n_active)
        dydt[:3*n_active] = vel.flatten()  # dx/dt = v
        dydt[3*n_active:] = accelerations.flatten()  # dv/dt = a
        
        return dydt
    
    # Initial state vector: [positions, velocities]
    current_state = np.concatenate([positions.flatten(), velocities.flatten()])
    current_time = 0.0
    
    # Setup for chunked integration
    dt_chunk = t_end / progress_steps
    
    # Calculate total number of frames based on simulation time and desired frame rate
    total_frames = max(11, int(t_end * frames_per_time_unit) + 1)  # Minimum 11 frames
    frames_per_chunk = max(2, total_frames // progress_steps)  # At least 2 points per chunk
    
    # Storage for results
    all_times = [0.0]
    all_trajectories = []
    all_velocities = []
    all_escaper_counts = []
    
    # Store initial state
    full_positions = np.full((n_stars, 3), np.nan)
    full_velocities = np.full((n_stars, 3), np.nan)
    full_positions[active_mask] = positions
    full_velocities[active_mask] = velocities
    all_trajectories.append(full_positions.copy())
    all_velocities.append(full_velocities.copy())
    all_escaper_counts.append(np.sum(active_mask))
    
    print(f"Starting chunked scipy integration...")
    print(f"Method: {method}, rtol: {rtol}, atol: {atol}")
    print(f"Integration time: 0 to {t_end} in {progress_steps} chunks")
    
    total_nfev = 0
    
    # Chunked integration loop
    for step in range(progress_steps):
        try:
            # Time span for this chunk
            t_span = (current_time, current_time + dt_chunk)
            
            # Number of evaluation points within this chunk based on frame rate
            # For the last chunk, adjust to ensure we hit exactly t_end
            if step == progress_steps - 1:
                # Last chunk - make sure we end exactly at t_end
                t_eval_chunk = np.linspace(t_span[0], t_end, frames_per_chunk)
            else:
                t_eval_chunk = np.linspace(t_span[0], t_span[1], frames_per_chunk)
            
            # Solve this chunk
            sol = solve_ivp(
                gravitational_ode, 
                t_span, 
                current_state,
                method=method,
                t_eval=t_eval_chunk,
                rtol=rtol,
                atol=atol,
                dense_output=False
            )
            
            if not sol.success:
                print(f"Warning: Integration failed at step {step+1} with message: {sol.message}")
                break
                
            total_nfev += sol.nfev
            
            # Update current state for next chunk
            current_state = sol.y[:, -1]
            current_time = sol.t[-1]
            
            # Process and store results (skip first point except for first chunk)
            start_idx = 1 if step > 0 else 0
            for i in range(start_idx, len(sol.t)):
                t = sol.t[i]
                n_active = np.sum(active_mask)
                
                if n_active > 0:
                    # Extract positions and velocities
                    pos_active = sol.y[:3*n_active, i].reshape(n_active, 3)
                    vel_active = sol.y[3*n_active:6*n_active, i].reshape(n_active, 3)
                else:
                    pos_active = np.empty((0, 3))
                    vel_active = np.empty((0, 3))
                
                # Store in full arrays with NaN for escaped stars
                full_positions = np.full((n_stars, 3), np.nan)
                full_velocities = np.full((n_stars, 3), np.nan)
                if n_active > 0:
                    full_positions[active_mask] = pos_active
                    full_velocities[active_mask] = vel_active
                
                all_trajectories.append(full_positions.copy())
                all_velocities.append(full_velocities.copy())
                all_escaper_counts.append(n_active)
                if i > 0 or step == 0:  # Don't duplicate time points
                    all_times.append(t)
            
            # Progress reporting
            progress = (step + 1) / progress_steps * 100
            print(f"  Integration progress: {progress:.1f}% (chunk {step+1}/{progress_steps}, {np.sum(active_mask)} stars)")
            
        except Exception as e:
            print(f"Integration failed at chunk {step+1}: {e}")
            break
    
    print("Integration complete!")
    print(f"Total function evaluations: {total_nfev}")
    print(f"Final star count: {all_escaper_counts[-1]} (started with {all_escaper_counts[0]})")
    
    # Convert to arrays
    traj_arr = np.array(all_trajectories)
    vel_arr = np.array(all_velocities)
    
    return traj_arr, vel_arr, masses, all_escaper_counts, np.array(all_times)


def calculate_total_energy(trajectory, velocities_history, masses, softening=0.1, G=1.0):
    """
    Calculate total energy (kinetic + potential) at each time step.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_steps+1, n_stars, 3)
        Positions at each time step
    velocities_history : ndarray, shape (n_steps+1, n_stars, 3)
        Velocities at each time step
    masses : ndarray, shape (n_stars,)
        Particle masses
    softening : float
        Gravitational softening length
    G : float
        Gravitational constant
        
    Returns
    -------
    times : ndarray
        Time points
    kinetic_energy : ndarray
        Kinetic energy at each time step
    potential_energy : ndarray
        Potential energy at each time step
    total_energy : ndarray
        Total energy at each time step
    """
    n_steps, n_stars, _ = trajectory.shape
    times = np.arange(n_steps)
    
    kinetic_energy = np.zeros(n_steps)
    potential_energy = np.zeros(n_steps)
    
    print("Calculating energy evolution...")
    
    for t in range(n_steps):
        if t % (n_steps // 10) == 0:
            print(f"Energy calculation: {100*t/n_steps:.0f}%", flush=True)
            
        pos = trajectory[t]
        vel = velocities_history[t]
        
        # Remove NaN values (escaped stars)
        valid_mask = ~np.isnan(pos[:, 0])
        pos_valid = pos[valid_mask]
        vel_valid = vel[valid_mask]
        n_valid = len(pos_valid)
        
        # Use appropriate number of masses for valid stars
        if n_valid > 0 and n_valid <= len(masses):
            masses_valid = masses[:n_valid]
            
            # Kinetic energy: 0.5 * m * v^2
            v_squared = np.sum(vel_valid**2, axis=1)
            kinetic_energy[t] = 0.5 * np.sum(masses_valid * v_squared)
            
            # Potential energy: -G * m1 * m2 / r (with softening)
            potential = 0.0
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    r_vec = pos_valid[i] - pos_valid[j]
                    r = np.sqrt(np.sum(r_vec**2) + softening**2)
                    potential -= G * masses_valid[i] * masses_valid[j] / r
            potential_energy[t] = potential
        else:
            kinetic_energy[t] = 0.0
            potential_energy[t] = 0.0
    
    total_energy = kinetic_energy + potential_energy
    
    return times, kinetic_energy, potential_energy, total_energy


def create_galaxy_animation(trajectory, save_filename='galaxy_evolution.gif'):
    """
    Create an animated movie of the galaxy evolution.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_steps+1, n_stars, 3)
        Positions at each time step
    save_filename : str
        Filename to save the animation
    """
    print(f"Creating animation: {save_filename}")
    
    n_steps, n_stars, _ = trajectory.shape
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Galaxy Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Initialize empty scatter plot
    scat = ax.scatter([], [], s=20, alpha=0.7, c='blue')
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        """Animation function for each frame."""
        pos = trajectory[frame]
        # Remove NaN values (escaped stars)
        valid_mask = ~np.isnan(pos[:, 0])
        pos_valid = pos[valid_mask]
        scat.set_offsets(pos_valid[:, :2])  # Use x, y coordinates
        time_text.set_text(f'Time Step: {frame}/{n_steps-1} ({len(pos_valid)} stars)')
        return scat, time_text
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=150, blit=True, repeat=True)
    
    # Save animation as GIF
    print("Saving animation (this may take a moment)...")
    ani.save(save_filename, writer='pillow', fps=7, dpi=80)
    plt.close(fig)
    
    print(f"Animation saved as: {save_filename}")


def create_galaxy_animation_3d(trajectory, save_filename='galaxy_evolution_3d.gif'):
    """
    Create a 3D animated movie of the galaxy evolution.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_steps+1, n_stars, 3)
        Positions at each time step
    save_filename : str
        Filename to save the 3D animation
    """
    print(f"Creating 3D animation: {save_filename}")
    
    n_steps, n_stars, _ = trajectory.shape
    
    # Set up the 3D figure and axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits based on trajectory data
    all_positions = trajectory.reshape(-1, 3)
    valid_positions = all_positions[~np.isnan(all_positions).any(axis=1)]
    
    if len(valid_positions) > 0:
        x_range = np.percentile(valid_positions[:, 0], [5, 95])
        y_range = np.percentile(valid_positions[:, 1], [5, 95])
        z_range = np.percentile(valid_positions[:, 2], [5, 95])
        
        # Add some padding
        padding = 0.2
        x_pad = (x_range[1] - x_range[0]) * padding
        y_pad = (y_range[1] - y_range[0]) * padding
        z_pad = (z_range[1] - z_range[0]) * padding
        
        ax.set_xlim(x_range[0] - x_pad, x_range[1] + x_pad)
        ax.set_ylim(y_range[0] - y_pad, y_range[1] + y_pad)
        ax.set_zlim(z_range[0] - z_pad, z_range[1] + z_pad)
    else:
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(-5, 5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Galaxy Evolution (3D View)')
    
    # Initialize empty scatter plot
    scat = ax.scatter([], [], [], s=30, alpha=0.7, c='blue')
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate_3d(frame):
        """Animation function for each frame."""
        pos = trajectory[frame]
        # Remove NaN values (escaped stars)
        valid_mask = ~np.isnan(pos[:, 0])
        pos_valid = pos[valid_mask]
        
        if len(pos_valid) > 0:
            # Update scatter plot data
            scat._offsets3d = (pos_valid[:, 0], pos_valid[:, 1], pos_valid[:, 2])
            
            # Color points by distance from center for visual appeal
            distances = np.sqrt(np.sum(pos_valid**2, axis=1))
            scat.set_array(distances)
        
        time_text.set_text(f'Time Step: {frame}/{n_steps-1} ({len(pos_valid)} stars)')
        
        return scat, time_text
    
    # Set a fixed, good viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate_3d, frames=n_steps, interval=200, blit=False, repeat=True)
    
    # Save animation as GIF
    print("Saving 3D animation (this may take a moment)...")
    ani.save(save_filename, writer='pillow', fps=5, dpi=80)
    plt.close(fig)
    
    print(f"3D Animation saved as: {save_filename}")


def plot_energy_conservation(times, kinetic_history, potential_history, total_history):
    """
    Create a plot showing energy conservation during the simulation.
    
    Parameters
    ----------
    times : array
        Time values
    kinetic_history : array
        Kinetic energy at each time step
    potential_history : array
        Potential energy at each time step  
    total_history : array
        Total energy at each time step
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Energy vs time
    plt.subplot(2, 1, 1)
    plt.plot(times, kinetic_history, label='Kinetic Energy', color='red')
    plt.plot(times, potential_history, label='Potential Energy', color='blue')
    plt.plot(times, total_history, label='Total Energy', color='black', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Conservation During Galaxy Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Energy change vs time
    plt.subplot(2, 1, 2)
    initial_total = total_history[0]
    energy_change = (total_history - initial_total) / abs(initial_total)
    plt.plot(times, energy_change, color='purple', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Relative Energy Change')
    plt.title('Relative Total Energy Change')
    plt.grid(True, alpha=0.3)
    plt.yscale('symlog', linthresh=1e-10)
    
    plt.tight_layout()
    plt.savefig('energy_conservation.png', dpi=150, bbox_inches='tight')
    #plt.show()


def main(
    n_stars=50,
    t_end=1.0,
    method='RK45',
    rtol=1e-3,
    atol=1e-6,
    theta=0.3,
    softening=0.1,
    galaxy_radius=10.0,
    remove_escapers=False,
    escaper_threshold=2.5,
    progress_steps=10,
    frames_per_time_unit=50
):
    """
    Main function to run the galaxy simulation.
    Parameters are now passed as arguments for flexibility.
    
    Parameters
    ----------
    n_stars : int
        Number of stars in the galaxy
    t_end : float
        Total simulation time
    method : str
        Integration method ('RK45', 'DOP853', 'Radau', 'BDF', 'LSODA')
    rtol : float
        Relative tolerance for the integrator
    atol : float
        Absolute tolerance for the integrator
    theta : float
        Barnes-Hut approximation parameter
    softening : float
        Gravitational softening parameter
    galaxy_radius : float
        Characteristic radius of the galaxy
    remove_escapers : bool
        If True, remove stars that escape beyond escaper_threshold * galaxy_radius
    escaper_threshold : float
        Multiplier for galaxy_radius to define escaper removal distance
    progress_steps : int
        Number of integration chunks for progress reporting (default: 10)
    frames_per_time_unit : int
        Number of animation frames per simulation time unit (default: 50)
    """
    print("=" * 60)
    print("Galaxy Simulation using bhut Barnes-Hut N-body Accelerator")
    print("=" * 60)

    print(f"\nSimulation Parameters:")
    print(f"  Number of stars: {n_stars}")
    print(f"  Integration method: {method}")
    print(f"  Simulation time: {t_end}")
    print(f"  Relative tolerance: {rtol}")
    print(f"  Absolute tolerance: {atol}")
    print(f"  Barnes-Hut theta: {theta}")
    print(f"  Gravitational softening: {softening}")
    print(f"  Galaxy radius: {galaxy_radius}")
    print(f"  Remove escapers: {remove_escapers}")
    if remove_escapers:
        print(f"  Escaper threshold: {escaper_threshold}")
    print()

    # Run simulation
    result = simulate_galaxy(
        n_stars=n_stars,
        t_end=t_end,
        theta=theta,
        softening=softening,
        galaxy_radius=galaxy_radius,
        remove_escapers=remove_escapers,
        escaper_threshold=escaper_threshold,
        method=method,
        rtol=rtol,
        atol=atol,
        progress_steps=progress_steps,
        frames_per_time_unit=frames_per_time_unit
    )
    trajectory, velocities_history, final_masses, escaper_counts, times = result

    # Use final masses from simulation (in case escapers were removed)
    masses = final_masses
    
    # Show energy conservation
    times_energy, kinetic_history, potential_history, total_history = calculate_total_energy(
        trajectory, velocities_history, masses
    )
    
    print(f"Final energy conservation check:")
    initial_energy = total_history[0]
    final_energy = total_history[-1]
    energy_change = abs(final_energy - initial_energy) / abs(initial_energy)
    print(f"  Initial total energy: {initial_energy:.6f}")
    print(f"  Final total energy: {final_energy:.6f}")
    print(f"  Relative energy change: {energy_change:.2e}")
    
    if remove_escapers:
        print(f"\nEscaper Statistics:")
        print(f"  Stars removed: {escaper_counts[-1]}")
        print(f"  Remaining stars: {n_stars - escaper_counts[-1]}")
    
    print("\nCreating animations...")
    
    # Create animations
    create_galaxy_animation(trajectory, save_filename="galaxy_2d_animation.gif")
    create_galaxy_animation_3d(trajectory, save_filename="galaxy_3d_animation.gif")
    
    # Create energy plot
    plot_energy_conservation(times_energy, kinetic_history, potential_history, total_history)
    
    print(f"\nSimulation completed successfully!")
    print(f"Animations saved as 'galaxy_2d_animation.gif' and 'galaxy_3d_animation.gif'")
    print(f"Energy plot saved as 'energy_conservation.png'")


if __name__ == "__main__":
    # Example usage: pass parameters here or use defaults
    main(
        n_stars=200,
        t_end=20.0,
        method='RK45',
        rtol=1e-3,
        atol=1e-6,
        theta=0.3,
        softening=0.1,
        galaxy_radius=10.0,
        remove_escapers=False,
        escaper_threshold=2.5,
        frames_per_time_unit=10
    )
