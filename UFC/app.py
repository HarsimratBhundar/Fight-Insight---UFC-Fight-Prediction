import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

from keras.models import load_model

import pandas as pd
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/fighters_clean.csv"))
df.set_index("NAME", inplace = True)

fighter_names = df.index.values.tolist()

fights = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/fights_clean.csv"))
fights.drop(["Fighter1", "Fighter2"], axis = 1, inplace = True)
X, y = fights.iloc[:, 2:], fights.iloc[:, 1]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

ridge_reg  = linear_model.Ridge(alpha = 0.5)
ridge_reg.fit(X_norm, y)

model = load_model(os.path.join(os.path.dirname(__file__), "cache.h5"))

def get_prediction(model, f1_name, f2_name):
    x1 = df.loc[f1_name, stats]
    x2 = df.loc[f2_name, stats]
    deltaX = (x1 - x2).values.reshape(1, -1)
    deltaX_norm = scaler.transform(deltaX)
    return round(model.predict(deltaX_norm)[0] * 100)

stats = ["SLPM", "SAPM", "STRA", "STRD", "TD", "TDA", "TDD", "SUBA"]

app = dash.Dash()

app.layout = html.Div(children=[html.Center(html.H1(children='Fight Insight', style={'font-family': 'Impact'})),
    
    html.Div(style={'width' : '40%' , 'float' : 'left', 'textAlign' : 'left'},
        children = [
            html.H3(children = "Fighter1"),
            html.Label('Select Fighter'),
            dcc.Dropdown(
                id = 'f1',
                options = [{'label' : i, 'value' : i} for i in fighter_names],
                value = fighter_names[0]
            ),
            html.Br(),
            html.Div(id = 'f1-info'),
            html.Div(id = 'f1-prob')
        ]),

    html.Div(style={'width' : '40%' , 'float' : 'right', 'textAlign' : 'left'},
        children = [
            html.H3(children = "Fighter2"),
            html.Label('Select Fighter'),
            dcc.Dropdown(
                id = 'f2',
                options = [{'label' : i, 'value' : i} for i in fighter_names],
                value = fighter_names[0]
            ),
            html.Br(),
            html.Div(id = 'f2-info'),
            html.Div(id = 'f2-prob')
        ]),
    html.Br(),
    html.Center(
        html.Button('Predict', id='predict-button', style={
                    'fontSize': '32px',
                    'backgroundColor': 'rgba(255,255,255,0.8)'
                })
        )
])

@app.callback(
    Output('f1-info', 'children'),
    [Input('f1', 'value')]
)

def set_f1_stats(f1_name):
    return [html.Label(children = 'Height: ' + df.loc[[f1_name]]['Height']), html.Br(),
    html.Label(children = 'Weight: ' + df.loc[[f1_name]]['Weight']), html.Br(),
    html.Label(children = 'Reach: ' + df.loc[[f1_name]]['REACH']), html.Br(),
    html.Label(children = 'Stance: ' + df.loc[[f1_name]]['Stance']), html.Br(),
    html.Label(children = 'D.O.B.: ' + df.loc[[f1_name]]['DOB']), html.Br()
    ]

@app.callback(
    Output('f2-info', 'children'),
    [Input('f2', 'value')]
)

def set_f2_stats(f2_name):
    return [html.Label(children = 'Height: ' + df.loc[[f2_name]]['Height']), html.Br(),
    html.Label(children = 'Weight: ' + df.loc[[f2_name]]['Weight']), html.Br(),
    html.Label(children = 'Reach: ' + df.loc[[f2_name]]['REACH']), html.Br(),
    html.Label(children = 'Stance: ' + df.loc[[f2_name]]['Stance']), html.Br(),
    html.Label(children = 'D.O.B.: ' + df.loc[[f2_name]]['DOB']), html.Br()
    ]

@app.callback(
     Output('f1-prob', 'children'),
     [Input('predict-button', 'n_clicks')],
      state=[State('f1', 'value'),
      State('f2', 'value')]

)

def get_f1_prob(n, f1_name, f2_name):
    if n > 0:
        return [html.Label(children = "Predicted chance of winning: " + str(get_prediction(ridge_reg, f1_name, f2_name)))]

@app.callback(
     Output('f2-prob', 'children'),
     [Input('predict-button', 'n_clicks')],
      state=[State('f1', 'value'),
      State('f2', 'value')]

)

def get_f2_prob(n, f1_name, f2_name):
    if n > 0:
        return [html.Label(children = "Predicted chance of winning: " + str(100 - get_prediction(ridge_reg, f1_name, f2_name)))]


if __name__ == '__main__':
    app.run_server(debug=True)
