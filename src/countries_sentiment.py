#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from os.path import join
import regex as re
import string
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from afinn import Afinn
import dateutil.parser as dparser
import spacy
from pandarallel import pandarallel
from spacy import displacy
from nltk.tokenize import word_tokenize
import requests
from json.decoder import JSONDecodeError
from multiprocessing.dummy import Pool as ThreadPool
import json
import plotly.express as px
import itertools
import plotly.graph_objects as go
import datetime
import os
import collections


def plot_nations_freq(df_nations):
    fig = px.area(df_nations, x='Date', y="Freq")
    fig.update_layout(
        yaxis_title="Frequency",
        xaxis_title="Date",
        title=dict(
            text=f'Frequency of countries by date',
            xanchor='center',
            x=0.5)
    )
    return fig


def plot_most_influential_countries(df_nations, top_entities):
    fig = go.Figure()

    for entity in top_entities["entity"]:
        y = []
        for x_date in df_nations["Date"]:
            nations = df_nations.query(f'Date=="{x_date}"')["Entities"].values[0]
            counts = collections.Counter(nations)
            y.append(counts[entity])
        fig.add_trace(go.Scatter(x=df_nations["Date"], y=y, name=entity))

    # Edit the layout
    fig.update_layout(
        yaxis_title="Frequency",
        xaxis_title="Date",
        title=dict(
            text=f'Frequency of the most polarized countries by date',
            xanchor='center',
            x=0.5, y=.9)
    )
    return fig


def plot_nations_geo(df_geo, freq_th=10):
    df_geo = df_geo[df_geo["freq"] > freq_th]
    fig = px.scatter_geo(df_geo, lat="lat", lon="long", hover_name="entity", size="freq",
                         color="norm", color_continuous_scale="Bluered_r")

    fig.update_layout(
        title=dict(
            text=f'Sentiment of the countries (Frequency threshold {freq_th})',
            xanchor='center',
            x=0.5, y=0.9)
    )
    # fig.update_traces(marker=dict(size=10))
    return fig


def plot_entitites_sentiment(df_entities, top_k=30):
    df_entities = df_entities.iloc[:top_k]
    fig = px.scatter(df_entities, x="score", y="freq", text="entity", size="freq",
                     color="score", color_continuous_scale="Bluered_r")
    fig.update_traces(textposition='top right')
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        title=dict(
            text=f'Sentiment of the countries (top {top_k} by frequency)',
            xanchor='center',
            x=0.5)
    )
    fig.update_traces(marker=dict(size=15), selector=dict(mode='markers'))
    return fig


def plot_entities_normalized(df_entities, freq_th=30):
    df_entities = df_entities[df_entities["freq"] > freq_th]
    df_entities_sub = df_entities.sort_values(by="norm", ascending=False)
    df_entities_sub.head()

    fig = px.bar(df_entities_sub, x="entity", y="norm", hover_data=["freq"],
                 color="norm", color_continuous_scale="Bluered_r")

    fig.update_layout(
        yaxis_title="Sentiment Score normalized",
        xaxis_title="Entity",
        title=dict(
            text=f'Sentiment of the countries normalized (Frequency threshold {freq_th})',
            xanchor='center',
            x=0.5),
        coloraxis_colorbar=dict(title="Sentiment")
    )
    return fig


def plot_entities_boxplot(df_entities, freq_th=2):
    df_entities = df_entities[df_entities["freq"] > freq_th]
    fig = px.box(df_entities, x="norm", points="all", hover_data=["entity", "freq"], height=400)
    fig.update_layout(
        xaxis_title="Sentiment Score normalized",
        title=dict(
            text=f'Boxplot of the normalized sentiment of the countries (frequency threshold {freq_th})',
            xanchor='center',
            x=0.5),
        coloraxis_colorbar=dict(title="Sentiment")
    )
    return fig


def get_most_influential_countries(df_ent, freq_th=2, top_k=6):
    df_ent = df_ent[df_ent["freq"]>freq_th]
    top_positive = df_ent.sort_values(by=["norm"], ascending=False).iloc[:top_k]
    top_positive = top_positive[top_positive["norm"] >= 0]
    top_negative = df_ent.sort_values(by=["norm"], ascending=True).iloc[:top_k]
    top_negative = top_negative[top_negative["norm"] < 0]

    return top_positive, top_negative


def main():
    sns.set_theme()
    pandarallel.initialize(progress_bar=False)

    # Constants
    DBPEDIA_ENDPOINT = "https://api.dbpedia-spotlight.org/en/annotate"
    POOL_SIZE = 30

    # Serialization folder
    SERIALIZATION_FOLDER = "pickle/"

    # Environment
    DATA_PATH = '../data/'
    EMAIL_DATA = 'Emails.csv'

    TOP_K = 30

    # Read dataframe

    # In[2]:

    df = pd.read_csv(join(DATA_PATH, EMAIL_DATA))

    # Subset useful fields
    df = df[['Id', 'SenderPersonId', 'MetadataDateSent',
             'ExtractedSubject', 'ExtractedBodyText']]
    # Preprocess dataframe data
    df = df.astype({'Id': int})
    df = df.set_index('Id')

    # Drop na values based on Extracted body
    len_before = len(df)
    df = df[df['ExtractedBodyText'].notna()]
    df = df[df['MetadataDateSent'].notna()]

    # Parse date
    df["MetadataDateSent"] = df["MetadataDateSent"].apply(lambda x: dparser.parse(x))
    df["DateYear"] = df["MetadataDateSent"].apply(lambda x: x.year)
    df["DateMonth"] = df["MetadataDateSent"].apply(lambda x: x.month)
    df["DateDay"] = df["MetadataDateSent"].apply(lambda x: x.day)

    print(f"Number of NA values in body and date: {len_before - len(df)}.\n"
          f"Number of valid email: {len(df)}")
    df.head()

    # Date and time distribution.
    #
    # Since the emails belong to Hillary Clinton's personal account instead of the institutional one, we should notice few emails sent during weekdays.

    # In[3]:

    def plot_days_dist(days_metadata):
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        days = [date.strftime("%A") for date in df[days_metadata.notnull()]["MetadataDateSent"]]
        sns.countplot(days, order=days_order)
        plt.xticks(rotation=45)
        plt.title("Days of the week distribution")
        plt.plot()

    plot_days_dist(df["MetadataDateSent"])

    # Preprocessing

    # In[4]:

    # Body preprocessing
    def clean_body(body):
        email_header = re.compile(r'.+[^<]+<[^>]+>', re.IGNORECASE)
        re_header = re.compile(r'(Delivered:\s+)?RE:[^\n]+\n', re.IGNORECASE)
        fw_header = re.compile(r'FW:[^\n]+\n', re.IGNORECASE)
        date_header = re.compile(r'\w+,\s\w+\s\d+,\s\d+[^\n]+\n', re.IGNORECASE)
        # Convert to lowercase
        body = body.lower()
        # Remove email
        body = re.sub(email_header, '', body).strip()
        # Remove "FW:"
        body = re.sub(fw_header, '', body).strip()
        # Remove "RE:"
        body = re.sub(re_header, '', body).strip()
        # Remove date
        body = re.sub(date_header, '', body).strip()
        return body

    sample = df['ExtractedBodyText'].loc[230]
    print('>>> Raw:\n' + sample)
    print('>>> Cleaned:\n' + clean_body(sample))

    # Drop the emails that are too short

    # In[5]:

    def tokenize_body(body):
        tokenized = word_tokenize(body)
        # Strip tokens
        tokenized = [token.strip() for token in tokenized]

        # --------- STRICT RULE ---------
        # Strict regex rule
        tokenized = [token for token in tokenized if re.match('\w+', token)]
        # --------- STRICT RULE ---------

        # Remove punctuation
        tokenized = [token for token in tokenized if token not in string.punctuation]
        # Remove stopwords
        stop = stopwords.words('english') + [':', '.', '@'] + ["n't"]
        tokenized = [token for token in tokenized if token not in stop]
        # Remove numbers
        tokenized = [token for token in tokenized if not re.search(r'\d', token)]
        return tokenized

    to_tokenize = clean_body(sample)
    tokenize_body(to_tokenize)

    # In[6]:

    def process_body(body):
        body = clean_body(body)
        tokenized = tokenize_body(body)
        return tokenized

    # In[7]:

    df['Tokenized'] = df['ExtractedBodyText'].parallel_apply(process_body)
    df.to_pickle(SERIALIZATION_FOLDER + "df.pkl")
    df.head()

    # In[8]:

    def plot_tokens_distribution(values):
        ax = sns.displot(values, log=True, bins=30)
        ax.set_xlabels("Number of tokens")
        plt.title("Number of tokens per email")
        plt.show()
        print("Describe tokens length:", values.describe())

    df['TokensLength'] = [len(x) for x in df['Tokenized']]

    plot_tokens_distribution(df['TokensLength'])

    # Wordcloud of the emails with few tokens

    # In[9]:

    TOKENS_THLD = 7

    def flatten_tokens(tokens_list):
        words_flatten = []
        for token in tokens_list:
            words_flatten += token
        return pd.Series(words_flatten)

    def plot_wordcloud(words_freq, title, max_font, max_words):
        wordcloud = WordCloud(max_font_size=max_font, max_words=max_words)
        wordcloud = wordcloud.generate_from_frequencies(words_freq)
        plt.figure()
        plt.title(title)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    # In[10]:

    words_flatten = flatten_tokens(df[df["TokensLength"] < TOKENS_THLD]["Tokenized"])
    title = f"Word cloud of the short emails (Num. of tokens < {TOKENS_THLD})"
    plot_wordcloud(words_flatten.value_counts(), title, 100, 70)

    # In[11]:

    words_flatten = flatten_tokens(df[df["TokensLength"] > TOKENS_THLD]["Tokenized"])
    title = f"Word cloud ot the long emails (Num. of tokens >= {TOKENS_THLD})"
    plot_wordcloud(words_flatten.value_counts(), title, 100, 70)

    # Named entity Recognition and linking on nations

    # In[12]:

    nlp = spacy.load("en_core_web_md")
    afn = Afinn()

    def extract_entities(email_body, target_label):
        entities = [ent for ent in nlp(email_body).ents if ent.label_ == target_label]
        return [(entity.text.lower(), afn.score(entity.sent.lemma_)) for entity in entities]

    # Slides demo

    # In[13]:

    def named_entity_recognition_demo():
        doc = """[...] The United States should immediately ask the Security Council to authorize a no-flight zone and make clear to Russia and China that if they block the resolution, the blood of the Libyan opposition will be on their hands. [...]"""
        displacy.render(nlp(doc), style="ent", jupyter=True)

    named_entity_recognition_demo()

    # In[14]:

    df["EntitiesGPE"] = df["ExtractedBodyText"].parallel_apply(extract_entities, target_label='GPE')
    df["EntitiesGPE"].head()

    # Aggregate (sum) the sentiment among emails.
    #
    # If an entity appears one or more times in the email, its frequency contribution is 1.

    # In[15]:

    def evaluate_entities(entities_list, order='freq'):
        entity_score = dict()
        freq_dict = dict()
        for entities in entities_list:
            entities_set = set()
            for entity, sentiment in entities:
                entity_score[entity] = entity_score.get(entity, 0) + sentiment
                entities_set.add(entity)
            for entity in entities_set:
                freq_dict[entity] = freq_dict.get(entity, 0) + 1
        out_dict = dict()
        for entity, frequnecy in freq_dict.items():
            out_dict[entity] = {
                "score": entity_score[entity],
                "freq": frequnecy,
                "norm": entity_score[entity] / frequnecy}
        return {k: v for k, v in sorted(out_dict.items(), key=lambda item: item[1][order], reverse=True)}

    def clean_entities(entities, blacklist):
        for target in map(lambda x: x.lower(), blacklist):
            if entities.get(target) is not None:
                del entities[target]
        return entities

    # In[16]:

    entities_to_remove = ['U.S.', 'US', 'the United States', 'America', 'usa',
                          "washington", "new york", "dc", "united states"]
    entities_gpe = evaluate_entities(df['EntitiesGPE'])
    entities_gpe = clean_entities(entities_gpe, entities_to_remove)
    print(list(entities_gpe.items())[:15])

    # Entity linking using DBpedia Spotlights

    # In[17]:

    def entity_linking(*args):
        entity, session = args
        params = {"text": entity, "confidence": 0.1}
        headers = {"accept": "application/json"}
        response_url = session.get(url=DBPEDIA_ENDPOINT, params=params, headers=headers)
        try:
            parsed = json.loads(response_url.content.decode())
            return parsed["Resources"][0]['@URI']
        except (IndexError, KeyError, JSONDecodeError):
            return None

    def resolve_entities(entities: dict) -> dict:
        with ThreadPool(POOL_SIZE) as pool:
            with requests.Session() as session:
                session_to_zip = itertools.repeat(session, len(entities))
                results = pool.starmap(entity_linking, zip(entities.keys(), session_to_zip))

        entities_resolved = entities
        for (entity, entity_dict), dbpedia in zip(entities.items(), results):
            entity_dict.update([('dbpedia', dbpedia)])
            entities_resolved[entity] = entity_dict
        return entities_resolved

    # In[18]:

    entities_gpe = resolve_entities(entities_gpe)
    print(list(entities_gpe.items())[:15])

    # Plot nations sentiment using

    # In[19]:

    def build_entities_df(entities):
        df_entities = pd.DataFrame(columns=["entity", "score", "freq"])
        for entity, entity_dict in entities.items():
            row = {"entity": entity, "score": entity_dict["score"],
                   "freq": entity_dict["freq"], "dbpedia": entity_dict["dbpedia"]}
            df_entities = df_entities.append(row, ignore_index=True)

        df_entities.sort_values(by=['freq'])
        df_entities = df_entities.astype(dict(freq=float))
        df_entities["norm"] = df_entities["score"] / df_entities["freq"]
        df_entities = df_entities.dropna()
        # False positive
        false_positive = ["mcchrystal", "mcdonough"]
        false_positive_ids = []
        for fp in false_positive:
            false_positive_ids.append(df_entities[df_entities["entity"] == fp].index)
        false_positive_ids = [x.values[0] for x in false_positive_ids]
        try:
            df_entities = df_entities.drop(false_positive_ids)
        except KeyError:
            pass

        return df_entities

    df_entities = build_entities_df(entities_gpe)
    df_entities.to_pickle(os.path.join(SERIALIZATION_FOLDER, "df_entities.pkl"))
    df_entities.head()

    # In[20]:

    plot_entities_boxplot(df_entities).show()

    # In[21]:

    plot_entitites_sentiment(df_entities, top_k=TOP_K).show()

    # In[22]:

    plot_entities_normalized(df_entities, top_k=TOP_K).show()

    # Plot on map

    # In[29]:

    from SPARQLWrapper import SPARQLWrapper, JSON
    import tqdm

    def dbpedia_get_geo(uri):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        query = r'select distinct ?lat ?long where {<' + uri + '> geo:lat ?lat. <' + uri + '> geo:long ?long.}'

        try:
            sparql.setQuery(query)
            # Convert results to JSON format
            sparql.setReturnFormat(JSON)
            result = sparql.query().convert()["results"]["bindings"][0]
            lat, long = float(result["lat"]["value"]), float(result["long"]["value"])
            return uri, lat, long
        except (KeyError, IndexError, TimeoutError):
            return uri, None, None

    def resolve_geo_info(df):
        df = df.dropna()
        dbpedia_uris = df["dbpedia"]
        with ThreadPool(POOL_SIZE) as pool:
            results = []
            for r in tqdm.tqdm(pool.imap_unordered(dbpedia_get_geo, dbpedia_uris), total=len(dbpedia_uris)):
                results.append(r)
            # results = pool.map(dbpedia_get_geo, dbpedia_uris)
        # Append geo info
        df_geo = pd.DataFrame(results, columns=["dbpedia", "lat", "long"])
        # df_geo = pd.concat([df, df_geo], axis=1)
        df = df.join(df_geo.set_index("dbpedia"), on='dbpedia')
        df = df.dropna()
        return df

    # In[30]:

    df_geo = resolve_geo_info(df_entities)
    df_geo.to_pickle(os.path.join(SERIALIZATION_FOLDER, "df_geo.pkl"))
    df_geo.head()

    plot_nations_geo(df_geo).show()

    # In[32]:

    fig = go.Figure(
        data=go.Choropleth(
            locations=df_geo['entity'],
            z=df_geo['score'] / df_geo['freq'],
            text=df_geo['entity'],
            # Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,
            # Blues,Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,Electric,Viridis,Cividis.
            colorscale='Bluered',
            autocolorscale=False,
            reversescale=True,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_tickprefix='',
            colorbar_title='Score',
        ))

    fig.update_layout(
        title_text='Sentiment associated to countries',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        annotations=[dict(
            x=0.55,
            y=0.1,
            xref='paper',
            yref='paper',
            showarrow=False,
            text="Hillary Clinton's emails"
        )]
    )

    fig.update_traces(locationmode="country names", selector=dict(type='choropleth'))

    fig.show()

    # Plot entities over time

    # In[33]:

    def parse_entities_name(entities):
        if entities:
            return [e[0] for e in entities]
        else:
            return []

    df["EntitiesGPEName"] = df["EntitiesGPE"].apply(parse_entities_name)

    # In[34]:

    df_nations = df.groupby(by=["DateYear", "DateMonth"]).apply(lambda x: list(x["EntitiesGPEName"]))
    df_nations = pd.DataFrame(df_nations, columns=["Entities"]).reset_index()
    df_nations["Entities"] = df_nations.reset_index()["Entities"].apply(lambda x: list(itertools.chain(*x)))
    df_nations["Entities"] = df_nations["Entities"].apply(lambda x: x if x != [] else None)
    df_nations = df_nations.dropna()
    df_nations["Date"] = [datetime.datetime(x["DateYear"], x["DateMonth"], 1) for _, x in df_nations.iterrows()]
    dx = pd.period_range(min(df_nations.Date), max(df_nations.Date), freq='M')
    dx = [datetime.datetime(x.year, x.month, 1) for x in dx]
    ts = pd.DataFrame(dx, columns=["Date"])
    df_nations = pd.merge(df_nations, ts, on="Date", how="outer").sort_values(by=["Date"])
    df_nations["Entities"] = df_nations["Entities"].apply(lambda x: x if type(x) is list else [])
    df_nations["Freq"] = df_nations["Entities"].apply(lambda x: len(x) if x != None else [])
    df_nations = df_nations.sort_values(by=["Date"])
    df_nations.to_pickle(os.path.join(SERIALIZATION_FOLDER, "df_nations.pkl"))
    df_nations.head()

    plot_nations_freq(df_nations).show()

    # In[36]:

    TOP_K_CATEGORY = 6
    FREQ_THLD = 10

    top_entities = get_most_influential_countries(df_entities, freq_th=FREQ_THLD, top_k=TOP_K_CATEGORY)
    top_entities

    # In[37]:

    df_nations.head()

    # In[38]:

    plot_most_influential_countries().show()

    # In[40]:

    df_entities.head()

    # In[ ]:


if __name__ == "__main__":
    main()
