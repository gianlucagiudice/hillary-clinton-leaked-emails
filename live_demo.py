import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import dash

import dash_core_components as dcc
import dash_html_components as html
from src.exploratory_analysis import *
from src.countries_sentiment import (
    plot_entities_boxplot, plot_entitites_sentiment, plot_entities_normalized,
    plot_nations_geo, plot_nations_freq, plot_most_influential_countries)

from dash.dependencies import Input, Output
TOP_K = 15

dataset = pd.read_csv("data/Emails.csv")
persons = pd.read_csv("data/Persons.csv")
df_entities = pd.read_pickle("pickle/df_entities.pkl")
df_geo = pd.read_pickle("pickle/df_geo.pkl")
df_nations = pd.read_pickle("pickle/df_nations.pkl")

df = preprocess_emails(dataset, persons)

top_people = df['SenderFullName'].value_counts(normalize=True)[:TOP_K]
cum_sum = np.cumsum(top_people)

chart2 = go.Figure()

chart2.add_trace(go.Bar(
    x=cum_sum.index[:TOP_K],
    y=top_people[:TOP_K],
    name="Markers and Text",
    marker_color="#636EFA",
    showlegend=False
))

chart2.add_trace(go.Scatter(
    x=cum_sum.index[:TOP_K],
    y=cum_sum,
    mode="text+lines+markers",
    # name = "Cumulata",
    text=[i for i in range(1, TOP_K + 1)],
    textposition="top center",
    showlegend=False
))
chart2.update_xaxes(
    tickangle=45
)
chart2.update_yaxes(
    title_text="Probabilità dell'attività",
)
chart2.update_layout(
    title=dict(
        text=f'Attività dei contatti',
        xanchor='center',
        x=0.5, y=.9),
)

chart3 = px.treemap(df, path=['SenderFullName'],
                    height=500,
                    title="Origine delle mail in percentuale")
chart3.update_layout(
    title=dict(
        x=0.5,
        xanchor='center')
)
chart3.update_traces(textinfo='label + percent parent')


def generate_hisplot_expl(dataframe):
    n_of_people = len(df["SenderFullName"].unique())

    i = 0
    x = 0
    while i < len(dataframe.value_counts(normalize=False)):
        for line in dataframe.value_counts(normalize=False):
            x += dataframe.value_counts(normalize=False)[i]
            i += 1

    return html.Div([
        dcc.Graph(
            id='displot-people-date',
            figure=px.histogram(dataframe, x="DateSent", marginal="rug", color="SenderFullName")),
        html.Center(html.H4(
            children=[f"I {len(dataframe.SenderFullName.unique())} contatti scelti, cioè il " +
                      f"{round(len(dataframe.SenderFullName.unique()) / n_of_people * 100, 2)}% del totale," +
                      f" sono responsabili per il {round(x / len(df) * 100, 2)}% del flusso totale di mail"]
        )
        )
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    children=[
        html.Div(
            style={
                #'background-image': "url(https://static.vecteezy.com/system/resources/previews/000/072/938/original/rays-background-vector.jpg)",
                'background-repeat': "no-repeat",
                'background-size': 'cover',
                'margin-right': "auto",
                'margin-left': 'auto',
                'opacity': 1,
                'float': 'none'
                },
            children=[
                html.Center(
                    html.H1(html.B("Hillary Clinton's Leaked Emails"))
                )
            ]
        ),

        dcc.Tabs([

            dcc.Tab(label='Analisi Esplorativa', children=[
                html.Center(html.H2(children='Attività dei contatti')),
                html.Br(),
                html.Center(html.H4(children='Nei grafici sottostanti sono esposte le attività dei contatti presenti ' +
                                             'nel dataset. Come è possibile notare, le attività cominciano il 2008-05-01 e finiscono ' +
                                             'il 2014-12-14, per una copertura totale di 2418 giorni, ovvero più di 6 anni.')),
                dcc.Dropdown(
                    id='dropdown',
                    options=[{'label': i, 'value': i} for i in top_people.index],
                    multi=True, placeholder='Selezionare i contatti desiderati... (Ordinati per utenti più attivi)'),
                html.Div(id='plot-container'),
                html.Br(),
                html.Div(
                    children=[
                        dcc.Graph(figure=chart3),
                        dcc.Graph(figure=chart2)
                    ]

                )]),

            dcc.Tab(label='Sentiment Analysis', children=[
                html.Div(
                    children=[
                        html.Center(html.H2(children="Sentiment analysis paesi"))
                    ]
                ),
                html.Div(
                    style={"marginTop": 75},
                    children=[
                        html.H3(children="Frequenza paesi con sentiment associato"),
                        html.Div(id="entities-sentiment-container"),
                        html.Div(
                            children=[
                                html.Label("Top k paesi:"),
                                dcc.Slider(
                                    id="slider-entities-sentiment", min=1, max=len(df_entities),
                                    step=1, value=30, marks={0: 0, len(df_entities): len(df_entities)}),
                            ]
                        ),

                    ]
                ),
                html.Div(
                    style={"marginTop": 75},
                    children=[
                        html.H3(children="Boxplot sentiment normalizzato"),
                        html.Div(id="boxplot-sentiment-container"),
                        html.Div(
                            children=[
                                html.Label("Threshold frequenza dei paesi:"),
                                dcc.Slider(
                                    id="slider-boxplot-sentiment", min=1, max=max(df_entities["freq"]),
                                    step=1, value=2, marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])}),
                            ]
                        ),

                    ]
                ),
                html.Div(
                    style={"marginTop": 75},
                    children=[
                        html.H3(children="Sentiment normalizzato dei paesi più frequenti"),
                        html.Div(id="sentiment-container"),
                        html.Div(
                            children=[
                                html.Label("Threshold frequenza dei paesi:"),
                                dcc.Slider(
                                    id="slider-sentiment", min=1, max=max(df_entities["freq"]),
                                    step=1, value=30, marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])}),
                            ]
                        ),
                    ]
                ),
                html.Div(
                    style={"marginTop": 75},
                    children=[
                        html.H3(children="Sentiment normalizzato dei paesi più frequenti"),
                        html.Div(id="sentiment-geo-container"),
                        html.Div(
                            children=[
                                html.Label("Threshold frequenza dei paesi:"),
                                dcc.Slider(
                                    id="slider-geo-sentiment", min=1, max=max(df_entities["freq"]),
                                    step=1, value=30, marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])}),
                            ]
                        ),
                    ]
                )
            ]),
        ])])


@app.callback(dash.dependencies.Output('plot-container', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_hisplot_expl(dropdown_value):
    if dropdown_value is None:
        return generate_hisplot_expl(df)
    dff = df[df.SenderFullName.str.contains('|'.join(dropdown_value))]
    return generate_hisplot_expl(dff)

@app.callback(Output('entities-sentiment-container', 'children'),
              Input('slider-entities-sentiment', 'value'))
def display_entities_sentiment_container(value):
    return dcc.Graph(figure=plot_entitites_sentiment(df_entities, top_k=value))

@app.callback(Output('sentiment-container', 'children'),
              Input('slider-sentiment', 'value'))
def display_entities_sentiment_container(value):
    return dcc.Graph(figure=plot_entities_normalized(df_entities, top_k=value))

@app.callback(Output('sentiment-geo-container', 'children'),
              Input('slider-geo-sentiment', 'value'))
def display_entities_sentiment_container(value):
    return dcc.Graph(figure=plot_nations_geo(df_geo, value))

@app.callback(Output('boxplot-sentiment-container', 'children'),
              Input('slider-boxplot-sentiment', 'value'))
def display_entities_sentiment_container(value):
    return dcc.Graph(figure=plot_entities_boxplot(df_entities, freq_th=value))


if __name__ == '__main__':
    app.run_server(debug=False, port=1346)
