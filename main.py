# main.py
import time
from config import MarketConfig, OptionConfig, SimConfig
from path_generator import PathGenerator
from pricer import OptionPricer

def run_simulation():
    print("--- Initializing RQMC Quant Engine ---")
    
    # 1. Define the Market State
    # We use 30% volatility to test the robustness of the engine
    market = MarketConfig(S0=100.0, r=0.05, sigma=0.30)  
    option = OptionConfig(K=100.0, T=1.0)
    
    # 2. Define the Engine Parameters
    # We use 50,000 paths and 52 weekly steps to mirror the academic setup
    sim = SimConfig(num_paths=65536, num_steps=52) 
    
    print(f"Market: S0={market.S0}, Volatility={market.sigma*100}%")
    print(f"Contract: Strike={option.K}, Expiration={option.T} Year(s)")
    print(f"Engine: {sim.num_paths:,} paths over {sim.num_steps} steps")
    print("Optimization: RQMC (Sobol) + Geometric Control Variate\n")
    
    start_time = time.time()
    
    # 3. Generate Optimized Paths (RQMC + Antithetic)
    print("Generating space-filling Sobol paths...")
    generator = PathGenerator(market, option, sim)
    
    # THE FIX: generate_paths() is called with no arguments 
    # because RQMC and Antithetic are now hardcoded into the physics engine.
    S_paths = generator.generate_paths()
    
    # 4. Pricing Logic (Control Variates)
    print("Evaluating contract payoffs & calculating variance reduction...\n")
    pricer = OptionPricer(market, option, sim)
    
    # Calculate Asian with the Geometric Control Variate
    asian_results = pricer.price_asian_control_variate(S_paths)
    
    print("Generating visualizations...")
    from visualizer import PathVisualizer
    PathVisualizer.plot_simulation(S_paths, option.K, num_paths_to_plot=200)
    
    execution_time = time.time() - start_time
    
    
    
    # 5. Display the Report
    print("==================================================")
    print("                 PRICING REPORT                   ")
    print("==================================================")
    print(f"Execution Time:          {execution_time:.4f} seconds")
    print(f"Optimization Stack:      RQMC + Geometric CV")
    print("--------------------------------------------------")
    print("ASIAN OPTION (Arithmetic Target):")
    print(f"  Raw Monte Carlo Price: £{asian_results['Raw Price']:.4f}")
    print(f"  Optimized (CV) Price:  £{asian_results['Optimized Price']:.4f}")
    print("--------------------------------------------------")
    print("COMPUTATIONAL METRICS:")
    print(f"  Raw Standard Error:    {asian_results['Raw SE']:.6f}")
    print(f"  Optimized CV SE:       {asian_results['CV SE']:.6f}")
    print(f"  Variance Reduced By:   {asian_results['Variance Reduction %']:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_simulation()