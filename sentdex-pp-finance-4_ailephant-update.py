## matplotlib.finance deprecated so use PLOTLY OFFLINE (successfully!)

import plotly.plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import pandas as pd
from datetime import datetime

df = pd.read_csv('tsla.csv')

trace = go.Ohlc(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])

layout = go.Layout(
    xaxis = dict(
        rangeslider = dict(
            visible = False
        )
    )
)

data = [trace]

fig = go.Figure(data=data,layout=layout)
plotly.offline.plot(fig, auto_open=True)