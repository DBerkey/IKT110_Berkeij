from django.shortcuts import render
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lookup table (hole diameter in cm and hole label)
HOLE_TABLE = {
    "A": 8,
    "B": 5,
    "C": 3.5,
    "D": 2.5,
    "E": 2,
    "F": 1,
    "G": 0.8,
    "H": 0.65,
    "I": 0.5,
    "J": 0.2
}

# Coverage modifiers with labels
COVERAGE_MODIFIERS = {
    "Full": 1.0,
    "Three-Quarter": 0.75,
    "Half": 0.5,
    "Quarter": 0.25
}

# Hole tolerances (cm)
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

def calculator_view(request):
    distance = None
    accuracy_data = None
    hole = None
    hole_coverage_label = "Full"
    object_size_val = ''
    hole_eye_distance_val = ''
    intergalactic = False

    if request.method == "POST":
        hole = request.POST.get("hole")
        object_size_str = request.POST.get("object_size", "")
        hole_eye_distance_str = request.POST.get("hole_eye_distance", "")
        coverage_modifier_label = request.POST.get("coverage_modifier", "Full")
        hole_coverage_modifier = COVERAGE_MODIFIERS.get(coverage_modifier_label, 1.0)
        hole_coverage_label = coverage_modifier_label
        intergalactic = request.POST.get("intergalactic") == "on"

        object_size = float(object_size_str) if object_size_str else 2
        hole_eye_distance = float(hole_eye_distance_str) if hole_eye_distance_str else 40
        object_size = object_size * 100  # convert meters to centimeters

        if intergalactic:
            distance = 42
        else:
            hole_cm = HOLE_TABLE.get(hole, 1)
            d_eff = hole_cm * hole_coverage_modifier

            # Distance formula
            distance = (object_size * hole_eye_distance) / (d_eff * 100)
            distance = round(distance, 2)

            # Error terms
            f = human_fraction(d_eff)
            delta_human = f * d_eff
            delta_cov = coverage_confusion(hole_cm)
            delta_hole = HOLE_TOLERANCE.get(hole, 0.05)

            # Nominal
            delta_nom = math.sqrt(delta_human**2 + delta_cov**2 + delta_hole**2)
            rel_nom = delta_nom / d_eff

            # Optimistic
            rel_opt = delta_hole / d_eff

            # Pessimistic
            delta_pess = delta_human + delta_cov + delta_hole
            rel_pess = delta_pess / d_eff

            accuracy_data = {
                "optimistic": {
                    "pct": round(rel_opt * 100, 1),
                    "abs": round(distance * rel_opt, 2)
                },
                "nominal": {
                    "pct": round(rel_nom * 100, 1),
                    "abs": round(distance * rel_nom, 2)
                },
                "pessimistic": {
                    "pct": round(rel_pess * 100, 1),
                    "abs": round(distance * rel_pess, 2)
                },
                "components": {
                    "human": round(delta_human, 3),
                    "coverage": round(delta_cov, 3),
                    "hole_tol": round(delta_hole, 3),
                    "d_eff": round(d_eff, 3)
                }
            }

        if object_size_str:
            object_size_val = object_size_str
        if hole_eye_distance_str:
            hole_eye_distance_val = hole_eye_distance_str

    return render(request, "calculator.html", {
        "holes": HOLE_TABLE.keys(),
        "coverage_modifiers": COVERAGE_MODIFIERS.keys(),
        "distance": distance,
        "hole": hole,
        "hole_coverage_modifier": hole_coverage_label,
        "object_size": object_size_val,
        "hole_eye_distance": hole_eye_distance_val,
        "intergalactic": intergalactic,
        "accuracy_data": accuracy_data,
    })

def review_view(request):
    return render(request, "review.html")

def manual_view(request):
    return render(request, "manual.html")

def sizeguide_view(request):
    return render(request, "sizeguide.html")
