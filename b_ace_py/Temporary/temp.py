import numpy as np
import matplotlib.pyplot as plt

# ---------- Problem setup ----------

def utility(theta, a):
    """u(theta, a) = -(theta - a)**2"""
    return -(theta - a) ** 2

def shifted_utility(theta, a, K=100.0):
    """
    Shifted utility u_tilde = K + u = K - (theta - a)**2.
    We require u_tilde >= 0 to define a valid density.
    """
    val = K - (theta - a) ** 2
    return max(val, 0.0)

def sample_theta_prior(size=1):
    """Sample from Exp(1) prior on theta."""
    return np.random.exponential(scale=1.0, size=size)

def sample_a_prior(size=1, a_min=0.0, a_max=4.0):
    """Uniform prior on a in [a_min, a_max]."""
    return np.random.uniform(a_min, a_max, size=size)

# ---------- APS via independence MH ----------

def aps_mh_sampler(
    n_samples=200_000,
    burn_in=20_000,
    K=100.0,
    a_min=0.0,
    a_max=4.0,
):
    # Initialize from proposal until shifted utility is positive
    theta = sample_theta_prior()[0]
    a = sample_a_prior(a_min=a_min, a_max=a_max)[0]
    ut = shifted_utility(theta, a, K=K)
    while ut <= 0.0:
        theta = sample_theta_prior()[0]
        a = sample_a_prior(a_min=a_min, a_max=a_max)[0]
        ut = shifted_utility(theta, a, K=K)

    samples_a = []
    accepted = 0
    total = n_samples + burn_in

    for i in range(total):
        theta_prop = sample_theta_prior()[0]
        a_prop = sample_a_prior(a_min=a_min, a_max=a_max)[0]
        ut_prop = shifted_utility(theta_prop, a_prop, K=K)

        if ut_prop > 0:
            # MH ratio simplifies because proposal = prior
            alpha = min(1.0, ut_prop / ut)
            if np.random.rand() < alpha:
                theta, a, ut = theta_prop, a_prop, ut_prop
                accepted += 1

        if i >= burn_in:
            samples_a.append(a)

    accept_rate = accepted / total
    return np.array(samples_a), accept_rate

# ---------- Run APS and estimate a* ----------

np.random.seed(155)   # for reproducibility
samples_a, acc_rate = aps_mh_sampler()

# Histogram-based marginal mode for a
bins = np.linspace(0.0, 4.0, 81)
hist, edges = np.histogram(samples_a, bins=bins)
max_bin_index = np.argmax(hist)
mode_estimate = 0.5 * (edges[max_bin_index] + edges[max_bin_index + 1])

print(f"Acceptance rate: {acc_rate:.3f}")
print(f"APS-estimated optimal action a* ≈ {mode_estimate:.3f}")

# ---------- Expected utility curve (analytic) ----------

# From earlier derivation: E[u(theta,a)] = -(1 + (1 - a)^2)
a_grid = np.linspace(0.0, 4.0, 400)
eu_vals = -(1.0 + (1.0 - a_grid) ** 2)

# ---------- Plot ----------

plt.figure()
plt.plot(a_grid, eu_vals, label="Expected utility E[u(θ, a)]")
plt.axvline(mode_estimate, linestyle="--",
            label=f"APS optimal a* ≈ {mode_estimate:.2f}")
plt.xlabel("Action a (bomb location)")
plt.ylabel("Expected utility")
plt.title("Expected Utility vs. Bomb Location with APS-Estimated Optimum")
plt.legend()
plt.grid(True)
plt.show()