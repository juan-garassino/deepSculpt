
import joblib

from sklearn.model_selection import GridSearchCV

from MinTrainer.data import get_data, clean_df, holdout
from MinTrainer.pipeline import create_pipeline
from MinTrainer.metrics import compute_rmse

from MinTrainer.mlflowbase import MLFlowBase

from MinTrainer.params import BUCKET_NAME

from google.cloud import storage


class Trainer(MLFlowBase):

    MODEL_STORAGE_FILENAME = 'model.joblib'

    def __init__(self, params):

        # base class parameters
        experiment_name = "[DE][Berlin][garassinoj]mintrainer"
        mlflow_server_uri = "https://mlflow.lewagon.ai/"

        # call the constructor of the base class
        super().__init__(experiment_name, mlflow_server_uri)

        self.params = params

    def train(self):

        # retrieve the data
        df = get_data()
        df = clean_df(df)
        X_train, X_val, y_train, y_val = holdout(df)

        self.X_val = X_val
        self.y_val = y_val

        # create the pipeline
        pipeline = create_pipeline()

        self.mlflow_create_run()
        self.mlflow_log_param("estimator", "randomforest")

        # option 1 : using a pipeline
        estimator = pipeline

        # store the estimator
        self.estimator = estimator

        # option 2 : using a gridsearch
        grid_search = GridSearchCV(
            pipeline,
            param_grid={
                'model__max_depth': [1, 2, 3],
                'model__min_samples_leaf': [1, 2]
            },
            cv=5
        )

        estimator = grid_search

        # train the pipeline
        print(X_train.columns)
        print(X_train.dtypes)
        print(X_train)
        estimator.fit(X_train, y_train)

        # store the estimator
        self.estimator = estimator.best_estimator_

        return self.estimator

    def predict(self):

        estimator = self.estimator
        X_val = self.X_val
        y_val = self.y_val

        y_pred = estimator.predict(X_val)

        print(f"pred: {y_pred}")

        rmse = compute_rmse(y_pred, y_val)

        print(f"rmse: {rmse}")

        self.mlflow_log_metric("rmse", rmse)

    def evaluate(self):
        pass

    def save_model(self):

        # save the model to disk
        joblib.dump(self.estimator, self.MODEL_STORAGE_FILENAME)

        # upload the model to gcp
        self.upload_model_to_gcp()

        print(self.params)

    def upload_model_to_gcp(self):

        storage_location = f'models/taxifare/recap4/{self.MODEL_STORAGE_FILENAME}'

        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(storage_location)

        blob.upload_from_filename(self.MODEL_STORAGE_FILENAME)


def run_trainer():

    # params for the training
    params = dict(
        estimator="randomforest"
    )

    # create a trainer object
    trainer = Trainer(params)

    # run the trainer
    trainer.train()

    # run the prediction
    trainer.predict()

    # save the model
    trainer.save_model()


if __name__ == '__main__':
    run_trainer()
