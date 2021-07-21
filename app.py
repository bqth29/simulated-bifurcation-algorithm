import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import dash_table

from data.data import assets, dates
from models.Markowitz import Markowitz

def format_date(date):

    return date

void = dbc.Row(html.P(" "))   

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])    
app.layout = dbc.Container(
    [
        void,
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id='dates-dropdown',
                        options=[
                            {
                                'label': format_date(date),
                                'value': date
                            } for date in dates
                        ],
                        placeholder = 'Select a date'
                    ),
                    width = 2
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='assets-dropdown',
                        options=[
                            {
                                'label': asset,
                                'value': asset
                            } for asset in assets
                        ],
                        placeholder = 'Select assets',
                        multi = True
                    ),
                    width = 7
                ),
                dbc.Col(
                    dbc.Input(
                        type="number", 
                        id = 'bits-input', 
                        min=1, 
                        step=1,
                        placeholder = "Number of bits"
                    ),
                    width = 2
                ),
                dbc.Col(
                    dbc.Button("Optimize", id='go-optimization', color="success", className="mr-1", n_clicks = 0),
                    width = 1
                )
            ]
        ),
        void,
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="pie-chart")
                    ],
                    width = 8
                ),
                dbc.Col(
                    [
                        dash_table.DataTable(
                            id='table',
                            columns = [
                                {
                                    'name': 'Asset',
                                    'id': 'assets'
                                },
                                {
                                    'name': 'Stocks',
                                    'id': 'stocks'
                                },
                                {
                                    'name': 'Ratio (%)',
                                    'id': 'ratios'
                                }
                            ],
                            style_cell=dict(textAlign='center'),
                            style_header=dict(backgroundColor="green", color="white"),
                            style_table={'width': '100%', 
                                    'height': '400px', 
                                    'overflowY': 'scroll', 
                                    'padding': '10px 10px 10px 20px'
                            }
                        )
                    ]
                )
            ]
        )
    ],
    fluid = False
)

@app.callback(
    Output('pie-chart', 'figure'),
    Output('table', 'data'),
    Input('go-optimization', 'n_clicks'),
    State('dates-dropdown', 'value'),
    State('assets-dropdown', 'value'),
    State('bits-input', 'value')
)

def plot_data(n_clicks, date, assets_list, bits):

    if n_clicks > 0:
        
        if date is not None and assets_list is not None and bits is not None:

            markowitz = Markowitz.from_csv(
                number_of_bits = bits,
                date = date,
                assets_list = assets_list
            )

            markowitz.optimize()
            
            return px.pie(markowitz.portfolio, values='stocks', names='assets'), markowitz.portfolio.to_dict('records')

    
    default = pd.DataFrame( 
            {
                'stocks': [100],
                'assets': ['_null'],
                'ratios': [100]
            }
        )
    
    return px.pie(
        default,
        values='stocks', 
        names='assets'
    ), default.to_dict('records')  

if __name__ == '__main__':
    app.run_server(debug=True)        