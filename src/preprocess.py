def clean_data(df):

    df = df[df["fare_amount"] > 0]

    df = df[
        (df["passenger_count"] > 0) &
        (df["passenger_count"] <= 6)
    ]

    df = df.dropna()

    df = df.drop_duplicates()

    return df