import numpy as np

rgb_energy_runs = [
    0.169884,
    0.172425,
    0.177140,
    0.173220,
    0.171373 
]
    
def calculate_mean_and_std(energy_runs):
    mean_energy = np.mean(energy_runs)
    std_energy = np.std(energy_runs, ddof=1)
    return mean_energy, std_energy

mean_energy, std_energy = calculate_mean_and_std(rgb_energy_runs)

print("Mean:", mean_energy)
print("Standard deviation:", std_energy)
