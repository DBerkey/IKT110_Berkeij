import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# HOLE DEFINITIONS (cm)
# --------------------------
HOLE_TABLE = {
    "A": 8.0,
    "B": 5.0,
    "C": 3.5,
    "D": 2.5,
    "E": 2.0,
    "F": 1.0,
    "G": 0.8,
    "H": 0.65,
    "I": 0.5,
    "J": 0.2,
}

COVERAGE_MODIFIER = 1.0  # Full coverage for baseline
OBJECT_SIZE = 200.0      # cm (2 m target)
HOLE_EYE_DISTANCE = 40.0 # cm

# --------------------------
# Error model
# --------------------------
HOLE_TOLERANCE = {
    "A": 0.05, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.05,
    "F": 0.02, "G": 0.02, "H": 0.02, "I": 0.02, "J": 0.02,
}

def human_fraction(d_eff_cm):
    if d_eff_cm >= 2.0: return 0.05
    if d_eff_cm >= 1.0: return 0.08
    if d_eff_cm >= 0.5: return 0.12
    if d_eff_cm >= 0.2: return 0.20
    return 0.40

def coverage_confusion(hole_cm):
    return 0.125 * hole_cm  # cm

def distance_for_hole(hole_cm, coverage=1.0):
    """Compute distance in meters for given hole & coverage"""
    return (OBJECT_SIZE * HOLE_EYE_DISTANCE) / (hole_cm * coverage) / 100.0

def uncertainties(hole_cm, coverage=1.0):
    """Return (nominal, optimistic, pessimistic) uncertainties in %"""
    d_eff = hole_cm * coverage
    dist_m = distance_for_hole(hole_cm, coverage)

    # Error components
    f = human_fraction(d_eff)
    delta_human = f * d_eff
    delta_cov = coverage_confusion(hole_cm)
    delta_hole = HOLE_TOLERANCE.get(hole_cm, 0.05)

    # --- Nominal (quadrature) ---
    delta_d_eff_nom = (delta_human**2 + delta_cov**2 + delta_hole**2) ** 0.5
    rel_nom = delta_d_eff_nom / d_eff

    # --- Optimistic (hole tolerance only) ---
    rel_opt = delta_hole / d_eff

    # --- Pessimistic (linear sum) ---
    delta_d_eff_pess = delta_human + delta_cov + delta_hole
    rel_pess = delta_d_eff_pess / d_eff

    return dist_m, rel_opt*100, rel_nom*100, rel_pess*100

# --------------------------
# Generate plot
# --------------------------
plt.figure(figsize=(12,7))

for hole_label, hole_cm in HOLE_TABLE.items():
    dist, opt, nom, pess = uncertainties(hole_cm, COVERAGE_MODIFIER)

    plt.scatter(dist, nom, color="blue", marker="o", s=80)
    plt.scatter(dist, opt, color="green", marker="^", s=60)
    plt.scatter(dist, pess, color="red", marker="v", s=60)

    # label nominal point
    plt.text(dist*1.02, nom, hole_label, fontsize=9)

plt.xlabel("Distance measured (m)")
plt.ylabel("Uncertainty (%)")
plt.title("Optimistic, Nominal, and Pessimistic Accuracy vs Distance (Full Coverage)")
plt.grid(True)
plt.xlim(0, 520)
plt.ylim(0, None)

plt.legend(["Nominal (realistic)", "Optimistic (best case)", "Pessimistic (worst case)"])
plt.show()
