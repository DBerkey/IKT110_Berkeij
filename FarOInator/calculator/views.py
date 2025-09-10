from django.shortcuts import render
import logging

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculator_view(request):
    distance = None
    hole = None
    object_size_val = ''
    hole_eye_distance_val = ''
    hole_coverage_modifier = 1.0  # default value
    hole_coverage_label = "Full"
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
            distance = (object_size * hole_eye_distance) / (HOLE_TABLE.get(hole, 1) * hole_coverage_modifier)
            distance = distance / 100 # convert to meters
            distance = round(distance, 2) # round to 2 decimal places
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
    })

def review_view(request):
    return render(request, "review.html")

def manual_view(request):
    return render(request, "manual.html")

def sizeguide_view(request):
    return render(request, "sizeguide.html")
