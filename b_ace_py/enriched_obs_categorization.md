# Enriched Observation Feature Categories

Your current 9 enriched features organized by source paper for ablation studies.

---

## Category 1: GEOMETRY (Apollonius Circle Theory)
**Source**: Weintraub, Pachter, Garcia - "An Introduction to Pursuit-Evasion Differential Games" (ACC 2020)

| Index | Feature Name | Range | Description |
|-------|--------------|-------|-------------|
| 0 | `time_to_capture` | [0, 1] | Normalized time for threat to intercept agent (lower = more urgent) |
| 1 | `capture_feasible` | {0, 1} | Binary: Can threat geometrically intercept agent? |

**Theoretical Basis**: 
- Apollonius circles define the locus of interception points given speed ratios
- Time-to-capture derived from pursuer/evader geometry and speed ratio μ = v_evader/v_pursuer
- Capture feasibility determined by whether agent lies within threat's Apollonius circle

**Enable Flag**: `enable_apollonius`

---

## Category 2: ENGAGEMENT (BEZ + DMC)
**Source**: Von Moll & Weintraub - "Basic Engagement Zones" (JAIS 2024) + "Reactive Vehicle Guidance using Dynamic Maneuvering Cue" (Draft)

| Index | Feature Name | Range | Description |
|-------|--------------|-------|-------------|
| 2 | `bez_penetration` | [-1, 1] | Normalized penetration into Basic Engagement Zone (+ = inside danger) |
| 3 | `inside_bez` | {0, 1} | Binary: Is agent inside threat's engagement zone? |
| 4 | `aspect_angle` | [-1, 1] | Agent's aspect angle relative to threat (normalized by π) |
| 5 | `dmc_normalized` | [-1, 1] | Dynamic Maneuvering Cue: degrees of turn needed to escape |
| 6 | `inside_threat` | {0, 1} | Binary: Is agent in threat zone requiring maneuver? |

**Theoretical Basis**:
- BEZ defines the region where a threat (weapon/pursuer) can intercept given its range R and speed ratio μ
- BEZ boundary: ρ(ξ) = μR[cos ξ + √(cos²ξ - 1 + (R+r)²/(μ²R²))]
- DMC = minimum heading change required to exit the BEZ
- Higher DMC = higher risk, DMC of 0° = safe, DMC of 180° = unavoidable capture

**Enable Flags**: `enable_bez`, `enable_dmc`

---

## Category 3: RANGE_LIMITED (Escape Feasibility + Offense)
**Source**: Weintraub, Von Moll, Pachter - "Range-Limited Pursuit-Evasion" (2023)

| Index | Feature Name | Range | Description |
|-------|--------------|-------|-------------|
| 7 | `evader_always_escapes` | {0, 1} | Binary: Pursuer's range insufficient to ever capture |
| 8 | `pursuer_always_captures` | {0, 1} | Binary: Pursuer's range guarantees capture regardless of evader heading |

**Additional Offensive Features** (computed using Range-Limited theory):

| Index | Feature Name | Range | Description |
|-------|--------------|-------|-------------|
| 9* | `offensive_dominance` | [-1, 1] | WEZ comparison: positive = agent has range advantage |
| 10* | `offensive_ttc_score` | [0, 1] | Offensive time-to-capture score |

*Only if `enable_offense_wez` and `enable_offense_ttc` are True

**Theoretical Basis**:
- Three regimes based on pursuer range R vs. Apollonius geometry:
  1. **Evader Always Escapes**: R < (d - μd)/(1-μ²) → Pursuer can't reach Apollonius circle
  2. **Pursuer Always Captures**: R ≥ (d-ρ)/(1-μ) → Pursuer envelops entire escape region
  3. **Mixed**: Some headings allow escape, others lead to capture
- Offensive metrics flip the perspective: agent as pursuer, threat as evader

**Enable Flags**: `enable_offense_wez`, `enable_offense_ttc` (escape feasibility always computed)

---

## Summary Table

| Category | Paper | Features | Count | Enable Flags |
|----------|-------|----------|-------|--------------|
| **GEOMETRY** | Weintraub et al. 2020 | time_to_capture, capture_feasible | 2 | `enable_apollonius` |
| **ENGAGEMENT** | Von Moll & Weintraub 2024 | bez_penetration, inside_bez, aspect_angle, dmc_normalized, inside_threat | 5 | `enable_bez`, `enable_dmc` |
| **RANGE_LIMITED** | Weintraub et al. 2023 | evader_always_escapes, pursuer_always_captures, (offensive_dominance, offensive_ttc_score) | 2-4 | `enable_offense_wez`, `enable_offense_ttc` |

**Total**: 9 features (with offense) or 7 features (without offense)

---

## Ablation Study Configuration

```python
# Full enriched observations (9 features)
feature_config = {
    'enable_apollonius': True,   # GEOMETRY: 2 features
    'enable_bez': True,          # ENGAGEMENT: 3 features
    'enable_dmc': True,          # ENGAGEMENT: 2 features
    'enable_offense_wez': True,  # RANGE_LIMITED: 1 feature
    'enable_offense_ttc': True,  # RANGE_LIMITED: 1 feature
    # escape_feasibility always on: 2 features
}

# Ablation: GEOMETRY only (2 features)
geometry_only = {
    'enable_apollonius': True,
    'enable_bez': False,
    'enable_dmc': False,
    'enable_offense_wez': False,
    'enable_offense_ttc': False,
}

# Ablation: ENGAGEMENT only (5 features)
engagement_only = {
    'enable_apollonius': False,
    'enable_bez': True,
    'enable_dmc': True,
    'enable_offense_wez': False,
    'enable_offense_ttc': False,
}

# Ablation: RANGE_LIMITED only (4 features)
range_limited_only = {
    'enable_apollonius': False,
    'enable_bez': False,
    'enable_dmc': False,
    'enable_offense_wez': True,
    'enable_offense_ttc': True,
    # escape_feasibility always computed: 2 features
}
```

---

## Feature Index Mapping (for reference)

When all features enabled, the enriched observation indices are:

```
Base observations: [0-26]  (27 dims from B-ACE)

Enriched features: [27-35] (9 dims)
  [27] time_to_capture        (GEOMETRY)
  [28] capture_feasible       (GEOMETRY)
  [29] bez_penetration        (ENGAGEMENT)
  [30] inside_bez             (ENGAGEMENT)
  [31] aspect_angle           (ENGAGEMENT)
  [32] dmc_normalized         (ENGAGEMENT)
  [33] inside_threat          (ENGAGEMENT)
  [34] offensive_dominance    (RANGE_LIMITED) - if enabled
  [35] offensive_ttc_score    (RANGE_LIMITED) - if enabled
  [34/36] evader_always_escapes    (RANGE_LIMITED)
  [35/37] pursuer_always_captures  (RANGE_LIMITED)
```

Note: Escape feasibility features (evader_always_escapes, pursuer_always_captures) are always appended last.
