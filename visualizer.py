# visualizer.py
import matplotlib.pyplot as plt
import numpy as np

class PathVisualizer:
    @staticmethod
    def plot_simulation(S_paths: np.ndarray, K: float, num_paths_to_plot: int = 100):
        """
        Plots the simulated stock paths and the terminal price distribution.
        S_paths shape: (num_steps + 1, num_paths)
        """
        # Create a figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        # --- Plot 1: The Parallel Universes (Paths) ---
        # We only plot a small subset (e.g., 100) so it doesn't crash your computer
        time_steps = np.arange(S_paths.shape[0])
        subset_paths = S_paths[:, :num_paths_to_plot]
        
        ax1.plot(time_steps, subset_paths, alpha=0.5, linewidth=0.8)
        ax1.axhline(y=K, color='black', linestyle='--', label=f'Strike Price (£{K})')
        ax1.set_title(f'Geometric Brownian Motion ({num_paths_to_plot} Sample Paths)')
        ax1.set_xlabel('Time Steps (Weeks)')
        ax1.set_ylabel('Stock Price (£)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: The Terminal Distribution ---
        # Look at the final prices of ALL 50,000 paths
        terminal_prices = S_paths[-1, :]
        
        ax2.hist(terminal_prices, bins=50, orientation='horizontal', color='skyblue', edgecolor='white')
        ax2.axhline(y=K, color='black', linestyle='--', label=f'Strike (£{K})')
        ax2.set_title('Distribution of Final Prices')
        ax2.set_xlabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()