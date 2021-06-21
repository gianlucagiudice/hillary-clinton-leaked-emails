import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import dash

import dash_core_components as dcc
import dash_html_components as html
from src.exploratory_analysis import *
from src.countries_sentiment import (
    plot_entities_boxplot, plot_entitites_sentiment, plot_entities_normalized, plot_nations_geo,
    plot_nations_freq, plot_most_influential_countries, get_most_influential_countries)

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

top_positive, top_negative = get_most_influential_countries(df_entities, freq_th=5, top_k=15)

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

div_style = {"marginTop": 50}
tab_style = {"marginLeft": 150, "marginRight": 150, "marginTop": 50}
slider_style = {"width": "60%", "margin": "0 auto"}


def get_checklist_options(df):
    return [
        {
            'label': f'{x["entity"]} '
                     f'(score: {round(x["norm"], 2)}) ,'
                     f'freq: {x["freq"]})',
            "value": x["entity"]
        }
        for _, x in df.iterrows()
    ]


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
        html.H5(
            children=[f"I {len(dataframe.SenderFullName.unique())} contatti scelti, cioè il " +
                      f"{round(len(dataframe.SenderFullName.unique()) / n_of_people * 100, 2)}% del totale," +
                      f" sono responsabili per il {round(x / len(df) * 100, 2)}% del flusso totale di mail"])
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    children=[
        html.Div(
            style={
                """
                'background-image': "url(https://static.vecteezy.com/system/
                resources/previews/000/072/938/original/rays-background-vector.jpg)",
                """

                'background-repeat': "no-repeat",
                'background-size': 'cover',
                'margin-right': "auto",
                'margin-left': 'auto',
                'opacity': 1,
                'float': 'none',
                'padding': 40
            },
            children=[
                html.Center(
                    html.H1(html.B("Hillary Clinton's Leaked Emails"))
                )
            ]
        ),

        dcc.Tabs([

            dcc.Tab(label='Analisi Esplorativa',
                    children=[
                        html.Div(
                            style=tab_style,
                            children=[
                                html.Center(html.H2(children='Attività dei contatti')),
                                html.Br(),
                                html.H5(
                                    children='Nei grafici sottostanti sono esposte le attività dei contatti '
                                             'presenti nel dataset. Come è possibile notare, le attività '
                                             'cominciano il 2008-05-01 e finiscono il 2014-12-14, per una '
                                             'copertura totale di 2418 giorni, ovvero più di 6 anni.'),
                                dcc.Dropdown(
                                    id='dropdown',
                                    options=[{'label': i, 'value': i} for i in top_people.index],
                                    multi=True, placeholder='Selezionare i contatti desiderati... '
                                                            '(Ordinati per utenti più attivi)'),
                                html.Div(id='plot-container'),
                                html.Br(),
                                html.Div(
                                    children=[
                                        dcc.Graph(figure=chart3),
                                        dcc.Graph(figure=chart2)
                                    ]

                                )

                            ]
                        ),
                    ]),

            dcc.Tab(label='Sentiment Analysis', children=[
                html.Div(
                    style=tab_style,
                    children=[
                        html.Div(
                            children=[
                                html.Center(html.H2(children="Sentiment analysis paesi"))
                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
                                html.H3(children="Frequenza paesi con sentiment associato"),
                                html.Div(id="entities-sentiment-container"),
                                html.Div(
                                    style=slider_style,
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
                            style=div_style,
                            children=[
                                html.H3(children="Boxplot sentiment normalizzato"),
                                html.Div(id="boxplot-sentiment-container"),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        html.Label("Threshold frequenza dei paesi:"),
                                        html.Center(
                                            dcc.Slider(
                                                id="slider-boxplot-sentiment", min=1, max=max(df_entities["freq"]),
                                                step=1, value=2,
                                                marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])})),
                                    ]
                                ),

                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
                                html.H3(children="Sentiment normalizzato dei paesi più frequenti"),
                                html.Div(id="sentiment-container"),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        html.Label("Threshold frequenza dei paesi:"),
                                        dcc.Slider(
                                            id="slider-sentiment", min=1, max=max(df_entities["freq"]),
                                            step=1, value=15,
                                            marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])}),
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
                                html.H3(children="Sentiment normalizzato dei paesi più frequenti"),
                                html.Div(id="sentiment-geo-container"),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        html.Label("Threshold frequenza dei paesi:"),
                                        dcc.Slider(
                                            id="slider-geo-sentiment", min=1, max=max(df_entities["freq"]),
                                            step=1, value=15,
                                            marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])}),
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
                                html.H3(children="Frequenza dei paesi nel tempo"),
                                html.Div(
                                    children=[
                                        dcc.Graph(figure=plot_nations_freq(df_nations))
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
                                html.H3(children=f"Frequenza dei paesi più influenti"),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        # html.Label(id="label-freq-influential"),
                                        html.Div(id="label-freq-influential"),
                                        dcc.Slider(
                                            id="slider-freq-th", min=1, max=max(df_entities["freq"]),
                                            step=1, value=5,
                                            marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])}),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            style={'display': 'inline-block', 'width': '49%'},
                                            children=[
                                                html.H5("Paesi più influenti sentiment positivo"),
                                                dcc.Checklist(
                                                    id="checklist-positive",
                                                    options=get_checklist_options(top_positive),
                                                    value=['israel']
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            style={'display': 'inline-block', 'width': '49%', 'position': 'absolute'},
                                            children=[
                                                html.H5("Paesi più influenti sentiment negativo"),
                                                dcc.Checklist(
                                                    id="checklist-negative",
                                                    options=get_checklist_options(top_positive),
                                                    value=['benghazi']
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                                html.Div(id="countries-freq"),
                            ]
                        )
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
    return dcc.Graph(figure=plot_entities_normalized(df_entities, freq_th=value))


@app.callback(Output('sentiment-geo-container', 'children'),
              Input('slider-geo-sentiment', 'value'))
def display_entities_sentiment_container(value):
    return dcc.Graph(figure=plot_nations_geo(df_geo, freq_th=value))


@app.callback(Output('boxplot-sentiment-container', 'children'),
              Input('slider-boxplot-sentiment', 'value'))
def display_entities_sentiment_container(value):
    return dcc.Graph(figure=plot_entities_boxplot(df_entities, freq_th=value))


# Frequency thld
@app.callback(Output('checklist-positive', 'options'),
              Input('slider-freq-th', 'value'))
def display_most_influential_countries(value):
    df = get_most_influential_countries(df_entities, freq_th=value, top_k=TOP_K)[0]
    return get_checklist_options(df)


@app.callback(Output('checklist-negative', 'options'),
              Input('slider-freq-th', 'value'))
def display_most_influential_countries(value):
    df = get_most_influential_countries(df_entities, freq_th=value, top_k=TOP_K)[1]
    return get_checklist_options(df)


@app.callback(Output('label-freq-influential', 'children'),
              Input('slider-freq-th', 'value'))
def display_most_influential_countries(value):
    return html.Label(f"Threshold frequenza dei paesi ({value}):")


@app.callback(Output('countries-freq', 'children'),
              [Input('checklist-positive', 'value'),
               Input('checklist-negative', 'value')])
def display_most_influential_countries(pos_ent, neg_ent):
    query = ' | '.join([f'entity == "{x}"' for x in pos_ent + neg_ent])
    if pos_ent + neg_ent:
        top_entities = df_entities.query(query)
        return dcc.Graph(figure=plot_most_influential_countries(df_nations, top_entities))


if __name__ == '__main__':
    app.run_server(debug=False, port=1346)
