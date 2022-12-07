# $WIPE_BEGIN
from datetime import datetime
import pytz

import pandas as pd

from deepCab.ml_logic.registry import load_model
from deepCab.ml_logic.preprocessor import preprocess_features
from deepCab.interface.main import pred

# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()

# $IMPLODE_BEGIN
@app.get("/predict")
def predict(
    pickup_datetime: datetime,  # 2013-07-06 17:18:00
    pickup_longitude: float,  # -73.950655
    pickup_latitude: float,  # 40.783282
    dropoff_longitude: float,  # -73.984365
    dropoff_latitude: float,  # 40.769802
    passenger_count: int,
):  # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """

    # ⚠️ if the timezone conversion was not handled here the user would be assumed to provide an UTC datetime

    # create datetime object from user provided date
    # pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user provided datetime with the NYC timezone
    eastern = pytz.timezone("US/Eastern")

    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # convert the user datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # format the datetime as expected by the pipeline
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # fixing a value for the key, unused by the model
    # in the future the key might be removed from the pipeline input
    # eventhough it is used as a parameter for the Kaggle submission
    key = "2013-07-06 17:18:00.000000119"

    # build X ⚠️ beware to the order of the parameters ⚠️
    X_pred = pd.DataFrame(
        dict(
            key=[key],  # useless but the pipeline requires it
            pickup_datetime=[formatted_pickup_datetime],
            pickup_longitude=[float(pickup_longitude)],
            pickup_latitude=[float(pickup_latitude)],
            dropoff_longitude=[float(dropoff_longitude)],
            dropoff_latitude=[float(dropoff_latitude)],
            passenger_count=[float(passenger_count)],
        )
    )

    # ⚠️ print output appears in the terminal running the `uvicorn` command

    # verify the order and data type of the columns (must be the exact same as during the training)
    # print(X_pred)
    # print(X_pred.columns)
    # print(X_pred.dtypes)

    # TODO(krokrob): cache the model

    # model = load_model()

    # X_processed = preprocess_features(X_pred)

    # y_pred = model.predict(X_processed)

    # y_pred = pred(X_pred)

    model = app.state.model

    X_processed = preprocess_features(X_pred)

    y_pred = model.predict(X_processed)

    # ⚠️ fastapi only accepts simple python data types as a return value
    # among which dict, list, str, int, float, bool
    # in order to be able to convert the api response to json
    return dict(fare=float(y_pred[0, 0]))


# $IMPLODE_END


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
