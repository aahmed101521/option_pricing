# Quantitative Option Pricing Engine: Asian Call Derivatives

A production-grade Python pricing engine for path-dependent derivatives (Arithmetic Asian Options). This project demonstrates the transition from standard mathematical modeling to highly optimized computational finance by leveraging advanced variance reduction techniques.

## Mathematical Architecture

Pricing an Arithmetic Asian Option presents a computational challenge: because the payoff relies on the arithmetic average of a log-normally distributed random walk (Geometric Brownian Motion), no exact closed-form solution exists. Brute-force Monte Carlo simulations converge at a slow $O(N^{-1/2})$ rate, requiring massive CPU overhead.

To solve this, this engine implements an optimized academic pricing stack:

1. **Randomized Quasi-Monte Carlo (RQMC):** Replaces standard pseudo-random number generation with Owen-scrambled **Sobol low-discrepancy sequences**, preventing dimensional clustering and accelerating convergence.
2. **Antithetic Variates:** Halves the required number of random draws and forces perfectly symmetric path distribution by mirroring the normal shocks ($+Z$ and $-Z$).
3. **Control Variates (Geometric Anchor):** Utilizes the highly-correlated Geometric Asian Option as a mathematical anchor. The engine calculates the exact closed-form Kemna-Vorst continuous approximation for the Geometric average, and uses the resulting delta to correct the Arithmetic Monte Carlo estimate, reducing variance by over 90%.

## Project Structure
* `src/config.py`: Data classes managing state (Market, Contract, and Simulation limits).
* `src/path_generator.py`: The stochastic physics engine generating vectorized GBM asset paths via `scipy.stats.qmc.Sobol`.
* `src/pricer.py`: The financial contract logic handling payoff evaluation, Black-Scholes Greeks calculation, and Control Variate scaling.
* `src/main.py`: The orchestrator script injecting dependencies and outputting the performance report.

## Quickstart

```bash
# Clone the repository
git clone [https://github.com/yourusername/asian-option-pricer.git](https://github.com/yourusername/asian-option-pricer.git)
cd asian-option-pricer

# Create a virtual environment & install requirements
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

# Run the simulation
python src/main.py