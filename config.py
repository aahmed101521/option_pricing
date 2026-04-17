#config.py

from dataclasses import dataclass

@dataclass 
class MarketConfig:
    S0: float = 100.0    #Initial Stock price
    r: float = 0.05     #Risk-free rate
    sigma: float = 0.2  #Volatility
    

@dataclass
class OptionConfig:
    K: float = 100.0    #Strike price
    T: float = 1.0      #Time to maturity (in years)
    
@dataclass
class SimConfig:
    num_paths: int = 10000  #Number of Monte Carlo paths
    num_steps: int = 252     #Number of time steps (e.g., daily steps for 1 year)        
    