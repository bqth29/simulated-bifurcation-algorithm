import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from plotly.offline import iplot
import plotly.graph_objs as go

def pie(portfolio, assets_list):

    """
    Draws a pie chart to show the assets allocation.
    """

    number_of_assets = len(portfolio)

    indexes = [i for i in range(number_of_assets) if portfolio[i] > 10**(-3)]
    assets = [portfolio[i] for i in indexes]
    names = [assets_list[i] for i in indexes]

    df = pd.DataFrame(
        {
            'value' : assets,
            'names': names
        }
    )

    fig = px.pie(df, values='value', names='names', title='Optimal portfolio')
    fig.show()

def table(portfolio, assets_list):

    """
    Draws a comprehensive table gathering all the data regarding the assets allocation.
    """

    number_of_assets = len(portfolio)

    indexes = [i for i in range(number_of_assets) if portfolio[i] > 10**(-3)]
    assets = [portfolio[i] for i in indexes]
    names = [assets_list[i] for i in indexes]

    df = pd.DataFrame(
        {
            'value' : [round(100*asset/sum(assets),5) for asset in assets],
            'stocks' : assets,
            'names': names
        }
    ).sort_values(by=['names'])
    
    trace = go.Table(
    header=dict(values=["Assets","Stocks to purchase",'Percentage of capital invested'],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[df.names,df.stocks,df.value],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

    data = [trace]
    iplot(data, filename = 'pandas_table')  