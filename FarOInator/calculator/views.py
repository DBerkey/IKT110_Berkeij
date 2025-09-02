from django.shortcuts import render

# Lookup table (hole diameter in mm -> distance in m for 2m target @ 40cm)
HOLE_TABLE = {
    2: 400,
    3: 267,
    4: 200,
    6: 133,
    9: 89,
    13: 62,
    19: 42,
    27: 30,
    40: 20,
    60: 13,
    85: 9,
    120: 7,
}

def calculator_view(request):
    distance = None
    hole = None
    if request.method == "POST":
        hole = int(request.POST.get("hole"))
        distance = HOLE_TABLE.get(hole, None)

    return render(request, "calculator.html", {
        "holes": HOLE_TABLE.keys(),
        "distance": distance,
        "hole": hole,
    })
