import pandas as pd


def preprocess_emails(dataset, persons):
    df = pd.DataFrame(dataset)

    df.rename(columns={"MetadataDateSent": "DateSent"}, inplace=True)
    df = pd.DataFrame(df[["SenderPersonId", "DateSent"]])
    df.dropna(inplace=True)

    df["DateSent"] = df["DateSent"].astype(str)
    df["DateSent"] = df["DateSent"].str.slice(0, 10, 1)
    df["DateSent"] = pd.to_datetime(df["DateSent"])

    df = df.join(persons.set_index('Id'), on='SenderPersonId')
    df = df.rename(columns={"Name": "SenderFullName"})
    df["SenderPersonId"] = df["SenderPersonId"].astype(str).str.slice(0, -2, 1)

    return df
