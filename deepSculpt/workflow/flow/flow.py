from deepCab.interface.main import preprocess, train, evaluate
from colorama import Fore, Style
from prefect import task, Flow, Parameter
from prefect.schedules import IntervalSchedule
import os
import requests
import datetime
from prefect.executors import LocalDaskExecutor


@task
def preprocess_new_data(experiment):
    """
    Run the preprocessing of the new data
    """
    preprocess()
    preprocess(source_type="val")


@task
def evaluate_production_model(status):
    """
    Run the `Production` stage evaluation on new data
    Returns `eval_mae`
    """
    eval_mae = evaluate()

    print(
        Fore.GREEN
        + "\n🔥 Ran task: EVAL PERF:"
        + Style.RESET_ALL
        + f"\n- Past model performance: {eval_mae}"
    )

    return eval_mae


@task
def re_train(status):
    """
    Run the training
    Returns train_mae
    """
    # preprocess data chunk by chunk?
    train_mae = train()

    print(
        Fore.GREEN
        + "\n🔥 Ran task: TRAIN MODEL:"
        + Style.RESET_ALL
        + f"\n- New model performance: {train_mae}"
    )

    return train_mae


@task
def notify(eval_mae, train_mae):
    base_url = "https://wagon-chat.herokuapp.com"
    channel = "johnini"
    url = f"{base_url}/{channel}/messages"
    author = "johnini"
    content = "Evaluation MAE: {} - New training MAE: {}".format(
        round(eval_mae, 2), round(train_mae, 2)
    )
    data = dict(author=author, content=content)
    response = requests.post(url, data=data)
    response.raise_for_status()
    print(
        Fore.GREEN
        + f"\n🔥 Run task: NOTIF"
        + Style.RESET_ALL
        + f"\n- Past performance: {eval_mae}"
        + f"\n- New performance: {train_mae}"
    )


def build_flow(schedule):
    """
    build the prefect workflow for the `taxifare` package
    """
    flow_name = os.environ.get("PREFECT_FLOW_NAME")

    with Flow(name=flow_name, schedule=schedule) as flow:

        # retrieve mlfow env params
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        # create workflow parameters
        experiment = Parameter(name="experiment", default=mlflow_experiment)

        # register tasks in the workflow
        status = preprocess_new_data(experiment)

        eval_mae = evaluate_production_model(status)

        train_mae = re_train(status)

        notify(eval_mae, train_mae)

    return flow


if __name__ == "__main__":

    # schedule = None
    schedule = IntervalSchedule(
        interval=datetime.timedelta(minutes=2), end_date=datetime.datetime(2022, 12, 1)
    )

    flow = build_flow(schedule)

    # flow.visualize()

    # flow.run()

    flow.executor = LocalDaskExecutor()

    flow.register(os.environ.get("PREFECT_FLOW_NAME"))
