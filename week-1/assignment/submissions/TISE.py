import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,5,500)

def simulate_wavefunction(x,E,V):
    if E>V:
        k = np.sqrt(2*(E-V))
        return np.cos(k*x)
    elif E<V:
        k = np.sqrt(2*(V-E))
        return np.exp(-1*k*x)   
    else:
        return np.ones(x.shape)

psi_A = simulate_wavefunction(x,12,2)#oscillating cos func
psi_B = simulate_wavefunction(x,2,7)#exponential decrement
psi_C = simulate_wavefunction(x,5,5)#horz line

plt.figure(figsize=(9,9))

plt.plot(x, psi_A, label="Case A: E=12, V=2")
plt.plot(x, psi_B, label="Case B: E=2, V=7")
plt.plot(x, psi_C, label="Case C: E=5, V=5")

plt.xlabel("Position x")
plt.ylabel("Wavefunction ψ(x)")
plt.title("Solutions of the 1D Time-Independent Schrödinger Equation")
plt.legend()
plt.grid(True)

plt.show()
        
    