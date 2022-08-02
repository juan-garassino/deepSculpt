
from sklearn.model_selection import train_test_split

from google.cloud import storage

import pandas as pd

from MinTrainer.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, MODEL_BASE_PATH


def get_data_from_gcp():

    data_file = 'data_train_1k.csv'

    client = storage.Client().bucket(BUCKET_NAME)

    blob = client.blob(BUCKET_TRAIN_DATA_PATH)

    blob.download_to_filename(data_file)

    df = pd.read_csv(data_file)

    return df


def get_data():

    # url = "s3://wagon-public-datasets/taxi-fare-train.csv"
    # df = pd.read_csv(url, nrows=100)

    df = get_data_from_gcp()

    print(df.shape)

    return df


def clean_df(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def holdout(df):

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    return (X_train, X_val, y_train, y_val)


if __name__ == '__main__':
    df = get_data()
    print(df)
