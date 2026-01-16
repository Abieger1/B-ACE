#!/usr/bin/env python3
"""
Repeated Prisoner's Dilemma with Reinforcement-Based Learning

Players:
- Two prisoners, each with two actions:
    0 = Confess (C)
    1 = Lie (L)

Payoffs (u1, u2):
- Both confess:        (-8, -8)
- P1 lies, P2 confesses: (-10,  0)
- P2 lies, P1 confesses: (  0, -10)
- Both lie:            ( -1, -1)

Reinforcement-learning model (simple Roth-Erev-style):
- Each player i has propensities q_i(a) for actions a ∈ {C, L}
- Initial propensities:
    q1(0) = (1, 2)
    q2(0) = (3, 5)
- At time t, probabilities are:
    p_i^t(a) = q_i^t(a) / sum_a q_i^t(a)
- Actions are sampled from these probabilities.
- After observing payoffs, we update:

    q_i^{t+1}(a) = (1 - phi) * q_i^t(a) + phi * r_i^t   if a was chosen
                 = (1 - phi) * q_i^t(a)                 if a was not chosen

  where r_i^t is a *positive* reinforcement derived from the payoff.

- Because your payoffs are negative, we shift them so reinforcement is positive:

    min_payoff = -10
    r_i^t = payoff_i^t - min_payoff + epsilon
          = payoff_i^t + 10 + epsilon

  With epsilon > 0 to avoid any zero reinforcement.
"""

import numpy as np

# --------------------
# Model parameters
# --------------------
NUM_STAGES = 100        # number of repeated plays (stage games)
phi = 0.8               # learning / forgetting parameter
epsilon = 0.1           # small positive constant so reinforcement is > 0
min_payoff = -10        # smallest payoff in the game (for shifting)

# Actions: 0 = Confess, 1 = Lie
CONFESS = 0
LIE = 1

# --------------------
# Payoff matrices
# --------------------
# U1[a1, a2] = payoff to Player 1 when P1 plays a1, P2 plays a2
# U2[a1, a2] = payoff to Player 2 when P1 plays a1, P2 plays a2
U1 = np.array([
    [-8,  0],   # P1 Confess vs (P2 Confess, P2 Lie)
    [-10, -1]   # P1 Lie     vs (P2 Confess, P2 Lie)
])

U2 = np.array([
    [-8, -10],  # P2 payoff when P1 Confess vs (P2 Confess, P2 Lie)
    [ 0,  -1]   # P2 payoff when P1 Lie     vs (P2 Confess, P2 Lie)
])

# --------------------
# Initial propensities
# --------------------
# q1(0) = (1, 2), q2(0) = (3, 5)
q1 = np.array([1.0, 2.0], dtype=float)  # Player 1: [C, L]
q2 = np.array([3.0, 5.0], dtype=float)  # Player 2: [C, L]

# For tracking results
history_actions = []
history_payoffs_1 = []
history_payoffs_2 = []

# Optional: track empirical frequency of (a1, a2) outcomes
# Keys: (a1, a2) tuples
outcome_counts = {
    (CONFESS, CONFESS): 0,
    (CONFESS, LIE): 0,
    (LIE, CONFESS): 0,
    (LIE, LIE): 0
}

# --------------------
# Simulation loop
# --------------------
for t in range(NUM_STAGES):
    # Convert propensities to choice probabilities
    p1 = q1 / q1.sum()
    p2 = q2 / q2.sum()

    # Sample actions according to these probabilities
    a1 = np.random.choice([CONFESS, LIE], p=p1)
    a2 = np.random.choice([CONFESS, LIE], p=p2)

    # Get payoffs
    payoff1 = U1[a1, a2]
    payoff2 = U2[a1, a2]

    # Store history
    history_actions.append((a1, a2))
    history_payoffs_1.append(payoff1)
    history_payoffs_2.append(payoff2)
    outcome_counts[(a1, a2)] += 1

    # Compute positive reinforcement signals
    r1 = payoff1 - min_payoff + epsilon  # = payoff1 + 10 + epsilon
    r2 = payoff2 - min_payoff + epsilon  # = payoff2 + 10 + epsilon

    # Update q1
    q1 = (1 - phi) * q1  # forgetting of all actions
    q1[a1] += phi * r1   # add reinforcement to chosen action

    # Update q2
    q2 = (1 - phi) * q2
    q2[a2] += phi * r2

# --------------------
# Summary statistics
# --------------------
history_payoffs_1 = np.array(history_payoffs_1)
history_payoffs_2 = np.array(history_payoffs_2)

avg_payoff_1 = history_payoffs_1.mean()
avg_payoff_2 = history_payoffs_2.mean()

# Final action probabilities
final_p1 = q1 / q1.sum()
final_p2 = q2 / q2.sum()

print("=== Repeated Prisoner's Dilemma with Reinforcement Learning ===")
print(f"Number of stages: {NUM_STAGES}")
print()
print("Final propensities:")
print(f"  Player 1 q: {q1}")
print(f"  Player 2 q: {q2}")
print()
print("Final action probabilities (approximate steady state):")
print(f"  Player 1: P(Confess) = {final_p1[CONFESS]:.3f}, P(Lie) = {final_p1[LIE]:.3f}")
print(f"  Player 2: P(Confess) = {final_p2[CONFESS]:.3f}, P(Lie) = {final_p2[LIE]:.3f}")
print()
print("Average payoffs over all stages:")
print(f"  Player 1: {avg_payoff_1:.3f}")
print(f"  Player 2: {avg_payoff_2:.3f}")
print()
print("Empirical frequencies of action profiles:")
for (a1, a2), count in outcome_counts.items():
    label = f"(P1={'C' if a1 == CONFESS else 'L'}, P2={'C' if a2 == CONFESS else 'L'})"
    print(f"  {label}: {count} times ({count / NUM_STAGES:.2%})")