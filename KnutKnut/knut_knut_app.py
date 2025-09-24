"""
Author: Urs Pfrommer
Date: 24-09-2025
"""
import pickle
import numpy as np
from flask import Flask, request
from makeModel import get_best_route

app = Flask(__name__)

# Load the updated models (dict with 4 routes as keys)
with open("KnutKnut/knut_knut_clean_models.pkl", "rb") as f:
    models = pickle.load(f)


def get_the_best_route_as_a_text_informatic(dep_hour, dep_min):
    dep_hour = int(dep_hour)
    dep_min = int(dep_min)

    best_route, best_time, predictions = get_best_route(dep_hour, dep_min, models)

    if best_route is None:
        return f"<p>No valid predictions for {dep_hour:02d}:{dep_min:02d}</p>"

    # Average travel time
    avg_time = np.mean([p[1] for p in predictions])
    time_saved = avg_time - best_time

    # Build table of all predictions
    rows = "".join(
        f"<tr><td>{route}</td><td>{time:.1f} minutes</td></tr>"
        for route, time in predictions
    )

    out = f"""
    <p>
    Departure time: {dep_hour:02d}:{dep_min:02d} <br>
    <b>Best travel route:</b> {best_route} <br>
    <b>Estimated travel time:</b> {best_time:.1f} minutes <br>
    <b>Time saved vs. average route:</b> {time_saved:.1f} minutes
    </p>
    <h4>Predicted travel times for all routes:</h4>
    <table border="1" cellpadding="4" cellspacing="0">
        <tr><th>Route</th><th>Predicted Time</th></tr>
        {rows}
    </table>
    <p><a href="/">Back</a></p>
    """
    return out


@app.route('/')
def get_departure_time():
    return """
        <h3>Knut Knut Transport AS</h3>
        <form action="/get_best_route" method="get">
            <label for="hour">Hour:</label>
            <select name="hour" id="hour">
                <option value="06">06</option>
                <option value="07">07</option>
                <option value="08">08</option>
                <option value="09">09</option>
                <option value="10">10</option>
                <option value="11">11</option>
                <option value="12">12</option>
                <option value="13">13</option>
                <option value="14">14</option>
                <option value="15">15</option>
                <option value="16">16</option>
            </select>
            
            <label for="mins">Mins:</label>
            <input type="text" name="mins" size="2"/>
            <input type="submit">
        </form>
    """


@app.route("/get_best_route")
def get_route():
    departure_h = request.args.get("hour")
    departure_m = request.args.get("mins")

    route_info = get_the_best_route_as_a_text_informatic(departure_h, departure_m)
    return route_info


if __name__ == "__main__":
    print("<starting>")
    app.run()
    print("<done>")

