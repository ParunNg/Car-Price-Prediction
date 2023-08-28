# Import packages
from dash import Dash, dcc, html, callback, Output, Input, State
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# paths of all components for car price predictions
model_path = "code/model/car_price_prediction.model"
scaler_path = 'code/preprocess/car_price_prediction.prep'
fuel_enc_path = 'code/preprocess/fuel_encoder.prep'
brand_enc_path = "code/preprocess/brand_encoder.prep"

# load all components
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
fuel_le = pickle.load(open(fuel_enc_path, 'rb'))
brand_ohe = pickle.load(open(brand_enc_path, 'rb'))

brand_cats = list(brand_ohe.categories_[0])
num_cols = ['max_power', 'year', 'engine']
default_vals = {'max_power': 82.4, 'year': 1248, 'engine': 2015}

# Create function for one-hot encoding a feature in dataframe 
def one_hot_transform(encoder, dataframe, feature):

    encoded = encoder.transform(dataframe[[feature]])

    # Transform encoded data arrays into dataframe where columns are based values
    categories = encoder.categories_[0]
    feature_df = pd.DataFrame(encoded.toarray(), columns=categories[1:])
    concat_dataframe = pd.concat([dataframe, feature_df], axis=1)
    
    return concat_dataframe.drop(feature, axis=1)

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
            dcc.Dropdown(id='fuel', options=['Diesel', 'Petrol'], value='Diesel'),

            dbc.Label("Brand"),
            dcc.Dropdown(id='brand', options=brand_cats, value=brand_cats[0]),

            html.Div([dbc.Button(id="submit", children="calculate selling price", color="primary"),

            html.Output(id="selling_price", children="", style={"margin-left": "5px"})
            ],
            style={"margin-top": "15px"})
        ],
        className="mb-3")
    ])

], fluid=True)

@callback(
    Output(component_id="selling_price", component_property="children"),
    State(component_id="max_power", component_property="value"),
    State(component_id="year", component_property="value"),
    State(component_id="engine", component_property="value"),
    State(component_id="fuel", component_property="value"),
    State(component_id="brand", component_property='value'),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)

def calculate_selling_price(max_power, year, engine, fuel, brand, submit):
    X = pd.DataFrame([[max_power, year, engine, fuel, brand]],
                     columns=['max_power', 'year', 'engine', 'fuel', 'brand'])
    
    for num_col in num_cols:
        if X[num_col][0] == None:
            X[num_col][0] = default_vals[num_col]
    
    X[num_cols] = scaler.transform(X[num_cols])
    X['fuel'] = fuel_le.transform(X['fuel'])
    X = one_hot_transform(brand_ohe, X, 'brand')
    y = np.round(np.exp(model.predict(X)), 2)

    return f"Selling price is: {y}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8001)