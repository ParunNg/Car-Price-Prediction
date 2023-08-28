# Import packages
from dash import Dash, html, callback, Output, Input, State
import pandas as pd
import pickle
import plotly.express as px
import dash_bootstrap_components as dbc

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# paths of all components for car price predictions
model_path = "model/car_price_prediction.model"
scaler_path = 'preprocess/car_price_prediction.prep'
fuel_enc_path = 'preprocess/fuel_encoder.prep'
brand_enc_path = "preprocess/brand_encoder.prep"

# load all components
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
fuel_le = pickle.load(open(fuel_enc_path, 'rb'))
brand_ohe = pickle.load(open(brand_enc_path, 'rb'))

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div([
            dbc.Label("Max Power"),
            dbc.Input(id="max_power", type="number", placeholder="Put a value for max power"),
            dbc.Label("Year"),
            dbc.Input(id="year", type="number", placeholder="Put a value for year"),
            dbc.Label("Engine"),
            dbc.Input(id="engine", type="number", placeholder="Put a value for engine"),
            dbc.Label("Fuel"),
            dbc.Input(id="fuel", type="number", placeholder="Put a value for fuel"),
            dbc.Button(id="submit", children="calculate y", color="primary", className="me-1"),
            dbc.Label("y is: "),
            html.Output(id="y", children="")
        ],
        className="mb-3")
    ])

], fluid=True)

@callback(
    Output(component_id="y", component_property="children"),
    State(component_id="x_1", component_property="value"),
    State(component_id="x_2", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def calculate_y(x_1, x_2, submit):
    return x_1 + x_2
# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8051)