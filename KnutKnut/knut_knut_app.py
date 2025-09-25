"""
Author: Douwe Berkeij & Urs Pfrommer
Date: 24-09-2025
AI Uses: For this file, OpenAI's ChatGPT Model 5 Auto was used to create the frontend of the 
app, which is everything from line 34 to line 206. 
The generated code was still manually reviewed, edited and tested to ensure functionality.
"""
import pickle
import numpy as np
from flask import Flask, request
from make_machine_learning_model import get_best_route

app = Flask(__name__)

# Load the updated models (dict with 4 routes as keys)
with open("KnutKnut/knut_knut_models_20250925_124950.pkl", "rb") as f:
    models = pickle.load(f)


def get_the_best_route_as_a_text_informatic(dep_hour, dep_min):
    dep_hour = int(dep_hour)
    if dep_min == '' or dep_min is None:
        dep_min = 0
    else:
        dep_min = int(dep_min)

    best_route, best_time, predictions = get_best_route(dep_hour, dep_min, models)

    if best_route is None:
        return f"<p>No valid predictions for {dep_hour:02d}:{dep_min:02d}</p>"

    # Average travel time
    avg_time = np.mean([p[1] for p in predictions])
    time_saved = avg_time - best_time

    # Build table of all predictions
    rows = "".join(
        f"<tr {'style=\"font-weight:bold; background:#e8f5e9;\"' if route == best_route else ''}>"
        f"<td>{route}</td><td>{time:.1f} minutes</td></tr>"
        for route, time in predictions
    )

    out = f"""
    <html>
    <head>
        <title>Knut Knut Transport AS</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f5f5f5;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 700px;
                margin: 40px auto;
                background: #fff;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            h3 {{
                text-align: center;
                color: #2c3e50;
            }}
            p {{
                font-size: 16px;
                line-height: 1.5;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th {{
                background: #2c3e50;
                color: white;
                padding: 8px;
                text-align: left;
            }}
            td {{
                padding: 8px;
            }}
            a {{
                display: inline-block;
                margin-top: 20px;
                text-decoration: none;
                color: white;
                background: #7f8c8d;
                padding: 10px 16px;
                border-radius: 6px;
                transition: background 0.3s;
            }}
            a:hover {{
                background: #606f70;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h3>Knut Knut Transport AS</h3>
            <p>
                <b>Departure time:</b> {dep_hour:02d}:{dep_min:02d} <br>
                <b>Best travel route:</b> {best_route} <br>
                <b>Estimated travel time:</b> {best_time:.1f} minutes <br>
                <b>Time saved vs. average route:</b> {time_saved:.1f} minutes
            </p>
            <h4>Predicted travel times for all routes:</h4>
            <table>
                <tr><th>Route</th><th>Predicted Time</th></tr>
                {rows}
            </table>
            <a href="/">← Back</a>
        </div>
    </body>
    </html>
    """
    return out


@app.route('/')
def get_departure_time():
    # Build hour options 06–22
    hour_options = ''.join(f'<option value="{h:02d}">{h:02d}</option>' for h in range(6, 23))

    html = """
    <html>
    <head>
        <title>Knut Knut Transport AS</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f5f5f5;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 500px;
                margin: 60px auto;
                background: #fff;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                text-align: center;
            }
            h3 {
                margin-bottom: 20px;
                color: #2c3e50;
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 15px;
                align-items: center;
            }
            label {
                font-weight: bold;
                margin-bottom: 5px;
            }
            select, input[type="number"], input[type="text"] {
                padding: 6px 10px;
                border: 1px solid #ccc;
                border-radius: 6px;
                font-size: 14px;
                width: 120px;
                text-align: center;
            }
            input[type="submit"] {
                background: #7f8c8d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.3s;
            }
            input[type="submit"]:hover {
                background: #606f70;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h3>Knut Knut Transport AS</h3>
            <form action="/get_best_route" method="get">
                <div>
                    <label for="hour">Hour:</label><br>
                    <select name="hour" id="hour">
                        {HOUR_OPTIONS}
                    </select>
                </div>
                <div>
                    <label for="mins">Mins:</label><br>
                    <input type="number" name="mins" id="mins" min="0" max="59" placeholder="00"/>
                </div>
                <input type="submit" value="Find Best Route">
            </form>
        </div>
    </body>
    </html>
    """
    return html.replace("{HOUR_OPTIONS}", hour_options)


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

