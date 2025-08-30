#!/usr/bin/env python3
"""
Galaxy Simulation using bhut Barnes-Hut N-body accelerator

This script simulates a galaxy of 1000 stars of equal mass using the
bhut package for efficient gravitational force calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
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


def simulate_galaxy(n_stars=100, n_steps=50, dt=0.01, theta=0.5, softening=0.1, galaxy_radius=10.0, remove_escapers=False, escaper_threshold=2.5):
    """
    Run a galaxy simulation using the Barnes-Hut algorithm.
    
    Parameters
    ----------
    n_stars : int
        Number of stars
    n_steps : int
        Number of time steps
    dt : float
        Time step size
    theta : float
        Barnes-Hut opening angle parameter
    softening : float
        Gravitational softening length
    galaxy_radius : float
        Characteristic radius of the galaxy (used for escaper removal)
    remove_escapers : bool
        If True, remove stars that escape beyond escaper_threshold * galaxy_radius
    escaper_threshold : float
        Multiplier for galaxy_radius to define escaper removal distance
        
    Returns
    -------
    trajectory : ndarray, shape (n_steps+1, N, 3)
        Positions at each time step (N may decrease if escapers are removed)
    velocities_history : ndarray, shape (n_steps+1, N, 3)
        Velocities at each time step
    masses : ndarray, shape (N,)
        Masses of stars (may decrease)
    escaper_counts : list
        Number of stars remaining at each time step
    """
    print(f"Initializing galaxy with {n_stars} stars...")
    
    # Generate initial conditions
    positions, velocities, masses = generate_galaxy_initial_conditions(n_stars, galaxy_radius)
    
    # Check virial equilibrium
    is_stable = check_virial_equilibrium(positions, velocities, masses, G=1.0, softening=softening)
    
    # Store trajectory and velocities for energy calculation
    trajectory = []
    velocities_history = []
    escaper_counts = []
    
    # Track which stars are still active (not escaped)
    active_mask = np.ones(n_stars, dtype=bool)  # All stars start active
    
    # Store initial conditions (all stars present)
    full_positions = np.full((n_stars, 3), np.nan)
    full_velocities = np.full((n_stars, 3), np.nan)
    full_positions[active_mask] = positions
    full_velocities[active_mask] = velocities
    trajectory.append(full_positions.copy())
    velocities_history.append(full_velocities.copy())
    escaper_counts.append(len(positions))
    
    # Create tree object for efficient updates
    tree = bhut.Tree(positions, masses, leaf_size=16, dim=3)  # Smaller leaf size for fewer stars
    tree.build()
    
    print("Starting simulation...")
    print(f"Time steps: {n_steps}, dt: {dt}")
    print(f"Barnes-Hut theta: {theta}, softening: {softening}")
    print("Progress: ", end="", flush=True)
    
    # Track tree rebuilds vs refits
    n_rebuilds = 0
    n_refits = 0
    
    # Main simulation loop using leapfrog integration
    for step in range(n_steps):
        if step % max(1, n_steps // 20) == 0:
            percent = 100 * step / n_steps
            print(f"{percent:.0f}%", end="... " if step < n_steps - 1 else "", flush=True)
        
        # Compute accelerations using Barnes-Hut algorithm
        accelerations = tree.accelerations(
            theta=theta,
            softening=softening,
            G=1.0
        )
        
        # Leapfrog integration
        # Update velocities (half step)
        velocities += 0.5 * dt * accelerations
        
        # Update positions (full step)
        positions += dt * velocities
        
        # Remove escapers if enabled
        if remove_escapers:
            radii = np.linalg.norm(positions, axis=1)
            keep_local = radii < escaper_threshold * galaxy_radius
            
            if not np.all(keep_local):
                # Update the global active mask
                current_active_indices = np.where(active_mask)[0]
                escaped_local_indices = np.where(~keep_local)[0]
                escaped_global_indices = current_active_indices[escaped_local_indices]
                active_mask[escaped_global_indices] = False
                
                # Keep only non-escaped stars for continued simulation
                positions = positions[keep_local]
                velocities = velocities[keep_local]
                active_masses = masses[active_mask]
                
                # Rebuild tree with remaining stars
                tree = bhut.Tree(positions, active_masses, leaf_size=16, dim=3)
                tree.build()
        
        # Update tree with new positions (refit for efficiency)
        if step < n_steps - 1:  # Don't refit on last step
            # Check if we should refit or rebuild
            try:
                tree.refit(positions)
                n_refits += 1
            except Exception:
                # If refit fails, rebuild the tree
                active_masses = masses[active_mask]
                tree.rebuild(positions, active_masses)
                n_rebuilds += 1
        
        # Update velocities (half step)
        accelerations = tree.accelerations(
            theta=theta,
            softening=softening,
            G=1.0
        )
        velocities += 0.5 * dt * accelerations
        
        # Store positions and velocities in full arrays
        full_positions = np.full((n_stars, 3), np.nan)
        full_velocities = np.full((n_stars, 3), np.nan)
        full_positions[active_mask] = positions
        full_velocities[active_mask] = velocities
        trajectory.append(full_positions.copy())
        velocities_history.append(full_velocities.copy())
        escaper_counts.append(len(positions))
    
    print("100% Complete!")
    print(f"Tree operations: {n_refits} refits, {n_rebuilds} rebuilds")
    if remove_escapers:
        print(f"Final star count: {escaper_counts[-1]} (started with {escaper_counts[0]})")
    print("Simulation complete!")
    
    # Convert lists to arrays - no padding needed now since all arrays are same size
    traj_arr = np.array(trajectory)
    vel_arr = np.array(velocities_history)
    
    return traj_arr, vel_arr, masses, escaper_counts


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


def analyze_energy_conservation(trajectory, velocities_history, masses, dt):
    """
    Analyze energy conservation and provide diagnostics.
    
    Parameters
    ----------
    trajectory : ndarray
        Position history
    velocities_history : ndarray
        Velocity history  
    masses : ndarray
        Particle masses
    dt : float
        Time step used
        
    Returns
    -------
    dict : Analysis results
    """
    times, kinetic, potential, total = calculate_total_energy(
        trajectory, velocities_history, masses
    )
    
    # Calculate various energy drift metrics
    energy_drift = total[-1] - total[0]
    
    # Handle divide by zero for relative drift calculation
    if abs(total[0]) > 1e-10:
        relative_drift = energy_drift / abs(total[0]) * 100
        max_excursion = np.max(np.abs(total - total[0])) / abs(total[0]) * 100
        systematic_component = np.mean(np.gradient(total)) * len(times) / abs(total[0]) * 100
    else:
        relative_drift = float('nan')
        max_excursion = float('nan') 
        systematic_component = float('nan')
    
    # Calculate energy drift rate
    drift_rate = energy_drift / (len(times) * dt)
    
    # Calculate maximum energy excursion (already handled above)
    # max_excursion = np.max(np.abs(total - total[0])) / abs(total[0]) * 100
    
    # Analyze if drift is systematic (monotonic) or oscillatory
    energy_gradient = np.gradient(total)
    # systematic_drift = np.mean(energy_gradient) * len(times) (already calculated above)
    
    analysis = {
        'total_drift_percent': relative_drift,
        'drift_rate': drift_rate,
        'max_excursion_percent': max_excursion,
        'systematic_component': systematic_component,
        'is_monotonic': np.all(energy_gradient > 0) or np.all(energy_gradient < 0)
    }
    
    return analysis


def suggest_improvements(analysis, dt, theta, softening):
    """
    Suggest parameter improvements based on energy analysis.
    """
    print(f"\n" + "="*50)
    print("ENERGY CONSERVATION ANALYSIS")
    print("="*50)
    
    # Handle NaN values for display
    drift_str = f"{analysis['total_drift_percent']:.2f}%" if not np.isnan(analysis['total_drift_percent']) else "N/A"
    excursion_str = f"{analysis['max_excursion_percent']:.2f}%" if not np.isnan(analysis['max_excursion_percent']) else "N/A"
    
    print(f"Total energy drift: {drift_str}")
    print(f"Maximum excursion: {excursion_str}")
    print(f"Drift is {'systematic' if analysis['is_monotonic'] else 'oscillatory'}")
    
    suggestions = []
    
    if abs(analysis['total_drift_percent']) > 0.5:
        if dt > 0.01:
            suggestions.append(f"• Reduce time step: current dt={dt:.3f}, try dt={dt*0.5:.3f}")
        
        if theta > 0.2:
            suggestions.append(f"• Use stricter Barnes-Hut criterion: current θ={theta}, try θ={theta*0.6:.1f}")
        
        if softening > 0.05:
            suggestions.append(f"• Reduce softening: current ε={softening:.2f}, try ε={softening*0.5:.2f}")
            
        if analysis['is_monotonic']:
            suggestions.append("• Systematic drift suggests integration error - try smaller dt")
        else:
            suggestions.append("• Oscillatory drift suggests force approximation errors - try smaller θ")
    
    if suggestions:
        print("\nSUGGESTED IMPROVEMENTS:")
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("\n✓ Energy conservation is acceptable (<0.5% drift)")
    
    print(f"\nCURRENT PARAMETERS:")
    print(f"  Time step (dt): {dt}")
    print(f"  Barnes-Hut θ: {theta}")
    print(f"  Softening: {softening}")


def plot_energy_evolution(times, kinetic_energy, potential_energy, total_energy, save_filename='galaxy_energy.png'):
    """
    Plot energy evolution and save to file.
    
    Parameters
    ----------
    times : ndarray
        Time points
    kinetic_energy : ndarray
        Kinetic energy at each time step
    potential_energy : ndarray
        Potential energy at each time step
    total_energy : ndarray
        Total energy at each time step
    save_filename : str
        Filename to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot individual energy components
    ax1.plot(times, kinetic_energy, 'r-', label='Kinetic Energy', linewidth=2)
    ax1.plot(times, potential_energy, 'b-', label='Potential Energy', linewidth=2)
    ax1.plot(times, total_energy, 'k--', label='Total Energy', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Energy')
    ax1.set_title('Galaxy Energy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot energy conservation (relative change in total energy)
    if abs(total_energy[0]) > 1e-10:
        energy_change = (total_energy - total_energy[0]) / abs(total_energy[0]) * 100
        ax2.plot(times, energy_change, 'g-', linewidth=2)
        ax2.set_ylabel('Total Energy Change (%)')
    else:
        # If initial energy is zero, plot absolute change instead
        energy_change = total_energy - total_energy[0]
        ax2.plot(times, energy_change, 'g-', linewidth=2)
        ax2.set_ylabel('Absolute Energy Change')
    ax2.set_xlabel('Time Step')
    if abs(total_energy[0]) > 1e-10:
        ax2.set_title('Energy Conservation (Relative Change in Total Energy)')
    else:
        ax2.set_title('Energy Conservation (Absolute Change in Total Energy)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot energy gradient (drift rate)
    energy_gradient = np.gradient(total_energy)
    ax3.plot(times[1:], energy_gradient[1:], 'purple', linewidth=1)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Energy Drift Rate')
    ax3.set_title('Energy Drift Rate (Gradient of Total Energy)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    #plt.show()
    
    print(f"Energy plot saved as: {save_filename}")


def track_escapers(trajectory, galaxy_radius):
    """
    Track the number of stars that escape beyond a threshold radius at each time step.
    Returns arrays of max radius, mean radius, and escaper count per time step.
    """
    n_steps, n_stars, _ = trajectory.shape
    max_radius = np.zeros(n_steps)
    mean_radius = np.zeros(n_steps)
    escaper_count = np.zeros(n_steps, dtype=int)
    threshold = 2.5 * galaxy_radius  # Escaper threshold
    for t in range(n_steps):
        pos = trajectory[t]
        radii = np.linalg.norm(pos, axis=1)
        max_radius[t] = np.max(radii)
        mean_radius[t] = np.mean(radii)
        escaper_count[t] = np.sum(radii > threshold)
    return max_radius, mean_radius, escaper_count


def plot_galaxy(trajectory, velocities_history, masses, n_stars=100, save_plots=True, dt=0.02, theta=0.5, softening=0.1, galaxy_radius=10.0, escaper_counts=None):
    """
    Create visualizations of the galaxy simulation.
    
    Parameters
    ----------
    trajectory : ndarray
        Position trajectory from simulation
    velocities_history : ndarray
        Velocity history from simulation
    masses : ndarray
        Particle masses
    n_stars : int
        Number of stars
    save_plots : bool
        Whether to save plots to files
    dt : float
        Time step used (for analysis)
    theta : float
        Barnes-Hut parameter used
    softening : float
        Softening parameter used
    """
    n_steps = trajectory.shape[0] - 1
    
    # Create animations
    create_galaxy_animation(trajectory, 'galaxy_evolution.gif')
    create_galaxy_animation_3d(trajectory, 'galaxy_evolution_3d.gif')
    
    # Calculate and plot energy evolution
    times, kinetic_energy, potential_energy, total_energy = calculate_total_energy(
        trajectory, velocities_history, masses
    )
    plot_energy_evolution(times, kinetic_energy, potential_energy, total_energy, 'galaxy_energy.png')
    
    # Analyze energy conservation
    analysis = analyze_energy_conservation(trajectory, velocities_history, masses, dt)
    suggest_improvements(analysis, dt, theta, softening)
    
    # Plot escaper counts if available
    if escaper_counts is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(escaper_counts, 'r-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Number of Stars')
        plt.title('Star Count Over Time')
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig('galaxy_star_count.png', dpi=150, bbox_inches='tight')
        #plt.show()
        print(f"Star count plot saved as: galaxy_star_count.png")
    
    # Additional 3D snapshot plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    final = trajectory[-1]
    # Remove NaN values for final plot
    valid_mask = ~np.isnan(final[:, 0])
    final_valid = final[valid_mask]
    ax.scatter(final_valid[:, 0], final_valid[:, 1], final_valid[:, 2], alpha=0.6, s=20)
    ax.set_title(f'3D View of Final Galaxy\n{len(final_valid)} Stars Remaining')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    if save_plots:
        plt.savefig('galaxy_3d.png', dpi=150, bbox_inches='tight')
    #plt.show()
    
    # Print energy statistics
    print(f"\nEnergy Statistics:")
    print(f"  Initial total energy: {total_energy[0]:.4f}")
    print(f"  Final total energy: {total_energy[-1]:.4f}")
    
    # Handle divide by zero for energy change calculation
    if abs(total_energy[0]) > 1e-10:  # Non-zero initial energy
        energy_change_percent = ((total_energy[-1] - total_energy[0])/abs(total_energy[0])*100)
        print(f"  Energy change: {energy_change_percent:.2f}%")
        
        # Energy conservation quality
        relative_change = abs((total_energy[-1] - total_energy[0])/total_energy[0])
        if relative_change < 0.001:
            quality = "Excellent"
        elif relative_change < 0.01:
            quality = "Good"
        else:
            quality = "Fair"
        print(f"  Energy conservation quality: {quality}")
    else:
        # Initial energy is zero (likely due to escaper removal or edge case)
        print(f"  Energy change: N/A (initial energy near zero)")
        print(f"  Energy conservation quality: N/A (cannot assess with zero initial energy)")
    
    # Track escapers
    max_radius, mean_radius, escaper_count = track_escapers(trajectory, galaxy_radius)
    plt.figure(figsize=(10, 6))
    plt.plot(max_radius, label='Max Radius')
    plt.plot(mean_radius, label='Mean Radius')
    plt.plot(escaper_count, label='Escaper Count (>2.5×radius)')
    plt.xlabel('Time Step')
    plt.ylabel('Radius / Count')
    plt.title('Galaxy Radius and Escaper Tracking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('galaxy_escapers.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print(f"Escaper plot saved as: galaxy_escapers.png")
    print(f"Max escaper count: {np.max(escaper_count)} at step {np.argmax(escaper_count)}")


def main():
    """
    Main function to run the galaxy simulation.
    """
    print("=" * 60)
    print("Galaxy Simulation using bhut Barnes-Hut N-body Accelerator")
    print("=" * 60)
    
    # Simulation parameters - optimized for energy conservation
    n_stars = 200
    n_steps = 1000  # More steps with smaller dt
    dt = 0.02     # Smaller time step for better energy conservation
    theta = 0.3   # Stricter Barnes-Hut criterion for better accuracy
    softening = 0.1  # Smaller softening for more realistic forces
    
    print(f"\nSimulation Parameters:")
    print(f"  Number of stars: {n_stars}")
    print(f"  Number of time steps: {n_steps}")
    print(f"  Time step: {dt}")
    print(f"  Barnes-Hut theta: {theta}")
    print(f"  Gravitational softening: {softening}")
    print()
    
    # Run simulation
    trajectory, velocities_history, final_masses, escaper_counts = simulate_galaxy(
        n_stars=n_stars,
        n_steps=n_steps,
        dt=dt,
        theta=theta,
        softening=softening,
        galaxy_radius=10.0,
        remove_escapers=False  # Set to True to enable escaper removal
    )
    
    # Use final masses from simulation (in case escapers were removed)
    masses = final_masses
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_galaxy(trajectory, velocities_history, masses, n_stars=n_stars, save_plots=True, 
                dt=dt, theta=theta, softening=softening, galaxy_radius=10.0, escaper_counts=escaper_counts)
    
    # Print some statistics
    initial_spread = np.std(trajectory[0], axis=0)
    final_spread = np.std(trajectory[-1], axis=0)
    
    print(f"\nSimulation Statistics:")
    print(f"  Initial position spread (std): X={initial_spread[0]:.2f}, Y={initial_spread[1]:.2f}, Z={initial_spread[2]:.2f}")
    print(f"  Final position spread (std):   X={final_spread[0]:.2f}, Y={final_spread[1]:.2f}, Z={final_spread[2]:.2f}")
    
    # Calculate center of mass drift (should be minimal)
    com_drift = np.linalg.norm(np.mean(trajectory[-1], axis=0) - np.mean(trajectory[0], axis=0))
    print(f"  Center of mass drift: {com_drift:.4f}")
    
    print("\nSimulation complete! Check the generated files:")
    print("  - galaxy_evolution.gif: Animated movie of galaxy evolution")
    print("  - galaxy_3d.png: 3D view of final galaxy")
    print("  - galaxy_energy.png: Total energy evolution")


if __name__ == "__main__":
    main()
