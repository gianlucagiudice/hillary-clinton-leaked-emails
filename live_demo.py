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
df_nations_metadata = pd.read_pickle("pickle/df_nations_metadata.pkl")

df = preprocess_emails(dataset, persons)

top_people = df['SenderFullName'].value_counts(normalize=True)[:TOP_K]
cum_sum = np.cumsum(top_people)

top_positive, top_negative = get_most_influential_countries(df_entities, df_nations_metadata, freq_th=10, top_k=15)

chart2 = go.Figure()

chart2.add_trace(go.Bar(
    x=cum_sum.index[:TOP_K],
    y=top_people[:TOP_K],
    name="Markers and Text",
    marker_color="#0a3a6e",
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
        font_size=25,
        xanchor='center',
        x=0.5, y=.9),
    # paper_bgcolor="#f2f7f7",
    font_size=15,
    yaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=0.1
    ),
    height=650
)

chart3 = px.treemap(df, path=['SenderFullName'],
                    height=500,
                    title="Origine delle mail in percentuale",
                    color_discrete_sequence=px.colors.sequential.tempo_r)
chart3.update_layout(
    title=dict(
        x=0.5,
        xanchor='center',
        font_size=25),
    # paper_bgcolor="#f2f7f7",
    font_size=20
)
chart3.update_traces(textinfo='label + percent parent')

div_style = {"marginTop": 50, "background-color": "#edf1f2"}
tab_style = {"marginLeft": 150, "marginRight": 150, "marginTop": 100, "marginBottom": 100,
             "color": "#d4a715", "background-color": "#edf1f2"}
slider_style = {"width": "60%", "margin": "0 auto"}



def get_checklist_options(df):
    return [
        {
            'label': f'{x["entity"]} '
                     f'(score: {round(x["norm"], 2)}, '
                     f'freq: {df_nations_metadata.loc[x["entity"]]["freq"]})',
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
    fig = px.histogram(dataframe, x="DateSent", marginal="rug", color="SenderFullName")
    fig.update_layout(
        xaxis_title="Data di invio",
    )
    return html.Div([
        dcc.Graph(
            id='displot-people-date',
            figure=fig),
        html.Center(html.H5(children=["I ",
                                html.Span(
                                  f"{len(dataframe.SenderFullName.unique())}", className="number-emphasis"),
                            " contatti scelti, cioè il ",
                              html.Span(
                                  f"{round(len(dataframe.SenderFullName.unique()) / n_of_people * 100, 2)}%", className="number-emphasis"),
                            " del totale, sono responsabili per il ",
                              html.Span(
                                  f"{round(x / len(df) * 100, 2)}%", className="number-emphasis"),
                            " del flusso totale di mail"]))
    ])


app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
                html.Div(
                    children=[
                    html.Center("HILLARY CLINTON'S LEAKED EMAILS", className="header"),
                    html.Div(
                        children=[
                            html.Div(
                                html.Img(src="https://welovetrump.com/wp-content/uploads/2019/11/hillary-1-1-768x431.jpg"),
                            style={"width": "100%","flex": "40%", "padding": "5px"}),
                            html.Div(
                                children=[
                                html.P("Durante il suo ruolo come Segretario di Stato degli Stati Uniti d’America,"
                                       " Hillary Clinton si è resa protagonista di uno scandalo riguardante l’utilizzo"
                                       " di server di mail privati per il trattamento di informazioni istituzionali, fra cui"
                                       " alcune classificate e confidenziali.", className="intro_style"),
                                html.P("In seguito all’istituzione delle indagini, le mail sono state rese completamente"
                                       " pubbliche da esponenti di WikiLeaks, aprendo la strada alla condanna di entrambi"
                                       " gli schieramenti politici statunitensi durante la campagna elettorale del 2016.", className="intro_style"),
                                html.P("Uno degli obiettivi di questo lavoro è l'analisi delle email trapelate per comprendere l'indirizzo di politica estera "
                                       "tenuto dalla Clinton e dalla sua rete di contatti.", className="intro_style")],
                            style={"width": "100%", "flex": "60%", "padding": "5px"})],
                    className="row")],
                ),
                html.Br(),

        dcc.Tabs([

            dcc.Tab(label='Analisi Esplorativa',
                    children=[
                        html.Div(
                            style=tab_style,
                            children=[
                                html.Center(html.H2(children='Attività dei contatti')),
                                html.H5(
                                    children=[
                                        html.Span("Hillary Clinton, Huma Abedin, Cheryl Mills e Jake Sullivan ", className="number-emphasis"),
                                        html.Span(
                                            children=[
                                                "sono i contatti più attivi e coprono circa il ",
                                                html.Span("75% del traffico", className="number-emphasis"), "."])]),
                                html.Br(),
                                dcc.Graph(figure=chart3),
                                html.Br(),
                                html.Div(
                                    children=[
                                        html.H5(
                                            children=['Nel grafico sottostante sono esposte le attività dei contatti '
                                                     'presenti nel dataset. Come è possibile notare, le attività ',
                                                      html.Span("cominciano il 2008-05-01 e finiscono il 2014-12-14 ", className="number-emphasis"),
                                                      'per una copertura totale di 2418 giorni, ovvero',
                                                      html.Span(" più di 6 anni", className="number-emphasis"), '.']),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id='dropdown', style={"color": "#0d5a6e"},
                                            options=[{'label': i, 'value': i} for i in top_people.index],
                                            multi=True, placeholder='Selezionare i contatti desiderati... '
                                                                    '(Ordinati per utenti più attivi)'),
                                        html.Div(id='plot-container'),
                                        html.Br(),
                                        html.Hr(),
                                        html.H5(
                                            children=[
                                                "In generale è possibile notare come le attività siano molto concentrate, soprattutto ",
                                                html.Span("fra i primi 15 contatti", className="number-emphasis"), "."]
                                        ),
                                        dcc.Graph(figure=chart2)

                                    ])]
                        ),
                    ],),

            dcc.Tab(label='Località',
                    children=[
                html.Div(
                    style=tab_style,
                    children=[
                        html.Div(
                            children=[
                                html.Center(html.H2(children="Sentiment analysis località"))
                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
                                html.H5(children=["Sono state identificate molte entità, per un totale di ",
                                        html.Span("860 località", className="number-emphasis"),". Tante località "
                                        "vengono nominate poche volte in maniera neutra, ma solo poche"
                                                 " scatenano reazioni vivaci, sia in positivo che in negativo."]),
                                html.H6("Usare lo slider per selezionare le località più frequenti"),
                                html.Br(),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        html.Label("Prime k località:"),
                                        dcc.Slider(
                                            id="slider-entities-sentiment", min=1, max=len(df_entities),
                                            step=1, value=30, marks={0: 0, len(df_entities): len(df_entities)}),
                                    ]
                                ),
                                html.Div(id="entities-sentiment-container"),

                            ]
                        ),
                        html.Br(),
                        html.Div(
                            style=div_style,
                            children=[
                                html.H5(children=["All'aumentare della frequenza con cui vengono nominate, possiamo notare come le località "
                                                 "cambino e acquistino maggiore o minore importanza relativa. In particolare ",
                                                  html.Span("i valori estremi cambiano di molto", className="number-emphasis"),
                                                 ". In questo modo si può avere un'idea più chiara di quali località hanno avuto un impatto più forte sulla politica"
                                                 " estera statunitense ",
                                                  html.Span("nel corso di tempi lunghi oppure in momenti ben precisi", className="number-emphasis"),
                                                  "."]),
                                html.Br(),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        html.Label("Threshold frequenza dei paesi:"),
                                        html.Center(
                                            dcc.Slider(
                                                id="slider-boxplot-sentiment", min=1, max=max(df_entities["freq"]),
                                                step=1, value=2,
                                                marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])})),
                                        html.Br(),
                                        html.Center(html.Div(id="boxplot-sentiment-container"))
                                    ]
                                ),

                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
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
                                html.Div(id="sentiment-container")
                            ]
                        ),
                        html.Div(
                            style=div_style,
                            children=[
                                    html.Div(
                                    style=slider_style,
                                    children=[
                                        html.Label("Threshold frequenza dei paesi:"),
                                        dcc.Slider(
                                            id="slider-geo-sentiment", min=1, max=max(df_entities["freq"]),
                                            step=1, value=15,
                                            marks={0: 0, max(df_entities["freq"]): max(df_entities["freq"])}),
                                html.Div(id="sentiment-geo-container")
                                    ]
                                ),
                            ]
                        ),

                        html.Div(
                            style=div_style,
                            children=[
                                html.H5(
                                    children=[
                                        "Nel tempo, varie località appaiono di particolare interesse, con ",
                                        html.Span("picchi positivi o negativi", className="number-emphasis"),
                                        " che indicano ",
                                        html.Span("avvenimenti di rilevanza internazionale ", className="number-emphasis"),
                                        "per la politica estera statunitense."]),
                                html.H6("Usare lo slider per selezionare le prime 10 località più importanti per soglia di occorrenza"),
                                html.Br(),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        # html.Label(id="label-freq-influential"),
                                        html.Div(id="label-freq-influential"),
                                        dcc.Slider(
                                            id="slider-freq-th", min=1, max=max(df_nations_metadata["freq"]),
                                            step=1, value=15,
                                            marks={0: 0, max(df_nations_metadata["freq"]): max(df_nations_metadata["freq"])}),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            style={'display': 'inline-block', 'width': '49%'},
                                            children=[
                                                html.H5("Località di maggior interesse con sentiment positivo"),
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
                                                html.H5("Località di maggior interesse con sentiment negativo"),
                                                dcc.Checklist(
                                                    id="checklist-negative",
                                                    options=get_checklist_options(top_positive),
                                                    value=['benghazi','haiti','israel','sri lanka']
                                                )
                                            ]
                                        ),
                                    ],
                                style={"text-align": "left"}),
                                html.Br(),
                                html.H6("Le frequenze per località sono espresse al primo del mese, ma sono calcolate su tutto quel mese."),
                                html.Div(id="countries-freq")
                            ]
                        )
                    ]
                )

            ]),

            dcc.Tab(label='Argomenti Discussi',
                children=[
                    html.Div(
                        style=tab_style,
                        children=[
                            html.Center(html.H2(children="sentiment analysis sui risultati del topic modeling")),
                            html.Div(
                                style=div_style,
                                    children= html.H5("Inizialmente, gli argomenti scambiati fra la Clinton e i suoi contatti appaiono vaghi, "
                                            "seppur riconducibili all'universo politico in generale.")),
                            html.Div(
                                children=[
                                    html.Div(
                                        html.H5(children=["Le ",
                                              html.Span("email più corte", className="number-emphasis"),
                                                          " (meno di 6 tokens) riguardano ",
                                              html.Span("situazioni d’ufficio e organizzative", className="number-emphasis"),
                                              ", così come il più evidente 'fyi' (For Your Information)."],
                                                className="cell2"
                                                ),
                                        className="columns2"),
                                    html.Div(
                                        html.H5(children=["Le ",
                                                          html.Span("email più lunghe", className="number-emphasis"),

                                                          " (più di 6 tokens) ",
                                                  html.Span("sono dichiaratamente politiche", className="number-emphasis"),
                                                  ", con 'state' come parola più utilizzata."],
                                                className="cell2"
                                                ),
                                       className="columns2")
                                ], className="row2"),
                            html.Div(
                                children=[
                                    html.Div(
                                        html.Img(src="/assets/word_cloud_short_clean.png", className="columns1"),
                                        className="imgContainer"
                                    ),
                                    html.Div(
                                        html.Img(src="/assets/word_cloud_long_clean.png", className="columns1"),
                                        className="imgContainer"
                                    )
                                ], className="row1"),
                            html.Hr(),
                            html.Div(
                                children=[
                                    html.Div(
                                        html.H5(children=["Attraverso l'utilizzo di ",
                                                          html.Span("tecniche di topic modeling ", className="number-emphasis"),
                                                          "sono stati estratti ",
                                                          html.Span("10 topics", className="number-emphasis"),
                                                          ", di cui ",
                                                          html.Span("5 positivi e 5 negativi", className="number-emphasis"),
                                                          ".",
                                                          ]

                                                ),
                                        ),
                                ], className="row2"),
                            html.Div(
                                children=[
                                    html.Div(
                                        html.H5(
                                              html.Span("Sentiment Positivo", className="number-emphasis"),
                                                className="cell2"
                                                ),
                                        className="columns2"),
                                    html.Div(
                                        html.H5(
                                                  html.Span("Sentiment Negativo", className="number-emphasis"),
                                                className="cell2"
                                                ),
                                       className="columns2")
                                ], style={"display":"flex"}),

                            # Topic 0
                            html.Div(children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S0-T0_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Internal Relations of the Party"),
                                                               "presid obama clinton"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S1-T0_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(
                                                children=[html.H3("China"),"china econom develop"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                )
                            ], className="row1"),
                            # Topic 1
                            html.Div(children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S0-T1_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Office Work"),
                                                               "work, call, thank, tomorrow"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S1-T1_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Industrial Support"),"waiver agreement case percent"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                )
                            ], className="row1"),
                            # Topic 2
                            html.Div(children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S0-T2_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Press"),
                                                               "secretary office meet"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S1-T2_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Afghanistan Conflict"),"afghanistan militari state war"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                )
                            ], className="row1"),
                            # Topic 3
                            html.Div(children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S0-T3_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Israeli-Palestinian Conflict"),
                                                               "israel palestine peace"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S1-T3_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("British Relations"),"tori cameron labour unionist"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                )
                            ], className="row1"),
                            # Topic 4
                            html.Div(children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S0-T4_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Women"),
                                                               "women secure support rights work"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            html.Img(
                                                src="/assets/topic_S1-T4_clean.png", className="columns1",
                                            ), className="wordcloudTopic"
                                        ),
                                        html.Div(
                                            html.Div(children=[html.H3("Against Republicans"),"republican koch boehner"]
                                            ), className="wordcloudDescription"
                                        ),
                                    ], className="topicContainer"
                                )
                            ], className="row1"),
                        ]
                    )
                ]
            )
        ])

    ])




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
    df = get_most_influential_countries(df_entities, df_nations_metadata, freq_th=value, top_k=TOP_K)[0]
    return get_checklist_options(df)


@app.callback(Output('checklist-negative', 'options'),
              Input('slider-freq-th', 'value'))
def display_most_influential_countries(value):
    df = get_most_influential_countries(df_entities, df_nations_metadata, freq_th=value, top_k=TOP_K)[1]
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
