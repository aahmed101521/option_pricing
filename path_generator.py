import numpy as np
from scipy.stats import qmc, norm
from config import MarketConfig, OptionConfig, SimConfig

class PathGenerator:
    def __init__(self, market: MarketConfig, option: OptionConfig, sim: SimConfig):
        self.market = market
        self.option = option
        self.sim = sim
        self.dt = self.option.T / self.sim.num_steps

    def generate_paths(self) -> np.ndarray:
        """
        Generates GBM paths using Randomized Quasi-Monte Carlo (RQMC) 
        and Antithetic Variates for maximum variance reduction.
        """
        half_paths = self.sim.num_paths // 2
        
        # 1. RQMC: Generate Sobol low-discrepancy sequence
        sampler = qmc.Sobol(d=self.sim.num_steps, scramble=True)
        # We need a matrix of uniform draws: (half_paths, num_steps)
        U = sampler.random(n=half_paths) 
        
        # 2. Inverse Transform: Convert uniform to Standard Normal
        # Transpose it so the shape is (num_steps, half_paths)
        Z_half = norm.ppf(U).T 
        
        # 3. Antithetic Mirroring
        Z = np.concatenate((Z_half, -Z_half), axis=1)
        
        # 4. GBM Physics
        S = np.zeros((self.sim.num_steps + 1, self.sim.num_paths))
        S[0] = self.market.S0
        
        drift = (self.market.r - 0.5 * self.market.sigma**2) * self.dt
        volatility = self.market.sigma * np.sqrt(self.dt)
        
        for t in range(1, self.sim.num_steps + 1):
            S[t] = S[t-1] * np.exp(drift + volatility * Z[t-1])
            
        return S