# pricer.py
import numpy as np
from scipy.stats import norm
from config import MarketConfig, OptionConfig, SimConfig

class OptionPricer:
    def __init__(self, market: MarketConfig, option: OptionConfig, sim: SimConfig):
        self.market = market
        self.option = option
        self.sim = sim
        self.discount_factor = np.exp(-self.market.r * self.option.T)

    def exact_geometric_asian(self) -> float:
        """Calculates the exact closed-form price of a discrete Geometric Asian Call."""
        S0, K, T, r, sigma = (self.market.S0, self.option.K, self.option.T, 
                              self.market.r, self.market.sigma)
        N = self.sim.num_steps
        
        sigma_G = sigma * np.sqrt((N + 1) * (2 * N + 1) / (6 * N**2))
        mu_G = (r - 0.5 * sigma**2) * (N + 1) / (2 * N) + 0.5 * sigma_G**2
        
        d1 = (np.log(S0 / K) + (mu_G + 0.5 * sigma_G**2) * T) / (sigma_G * np.sqrt(T))
        d2 = d1 - sigma_G * np.sqrt(T)
        
        return np.exp(-r * T) * (S0 * np.exp(mu_G * T) * norm.cdf(d1) - K * norm.cdf(d2))

    def evaluate_asian_payoffs(self, S: np.ndarray):
        """Calculates both Arithmetic and Geometric simulated payoffs."""
        A_avg = np.mean(S[1:], axis=0) 
        arithmetic_payoffs = np.maximum(A_avg - self.option.K, 0) * self.discount_factor
        
        G_avg = np.exp(np.mean(np.log(S[1:]), axis=0))
        geometric_payoffs = np.maximum(G_avg - self.option.K, 0) * self.discount_factor
        
        return geometric_payoffs, arithmetic_payoffs

    def price_asian_control_variate(self, S: np.ndarray) -> dict:
        """Prices the Arithmetic Asian using the Geometric Asian as the CV."""
        X, Y = self.evaluate_asian_payoffs(S) 
        E_X = self.exact_geometric_asian()    
        
        cov_matrix = np.cov(X, Y)
        c = cov_matrix[0, 1] / cov_matrix[0, 0] 
        
        Y_cv = Y - c * (X - E_X)
        
        num_paths = len(Y)
        raw_se = np.std(Y) / np.sqrt(num_paths)
        cv_se = np.std(Y_cv) / np.sqrt(num_paths)
        
        return {
            "Raw Price": np.mean(Y),
            "Optimized Price": np.mean(Y_cv),
            "Raw SE": raw_se,
            "CV SE": cv_se,
            "Variance Reduction %": (1 - (cv_se**2 / raw_se**2)) * 100
        }