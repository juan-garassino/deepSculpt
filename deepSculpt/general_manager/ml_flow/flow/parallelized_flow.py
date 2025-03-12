import os
from prefect import task, Flow, Parameter
from deepCab.interface.main import preprocess, train, evaluate
from deepCab.flow.flow import notify


@task
def preprocess_new_train(experiment):
    """
    Run the preprocessing of the train_new data
    """
    preprocess()


@task
def preprocess_new_val(experiment):
    """
    Run the preprocessing of the val_new data
    """
    preprocess(source_type="val")


@task
def evaluate_production_model(preproc_train_status, preproc_val_status):
    """
    Run the `Production` stage evaluation on new data
    Returns `eval_mae`
    """
    eval_mae = evaluate()
    return eval_mae


@task
def re_train(preproc_train_status, preproc_val_status):
    """
    Run the training
    Returns train_mae
    """
    train_mae = train()
    return train_mae


def build_parallel_flow():
    """
    build the prefect workflow for the `taxifare` package
    """
    flow_name = os.environ.get("PREFECT_FLOW_NAME")

    with Flow(flow_name) as flow:

        # retrieve mlfow env params
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        # create workflow parameters
        experiment = Parameter(name="experiment", default=mlflow_experiment)

        # register tasks in the workflow
        preproc_train_status = preprocess_new_train(experiment)
        preproc_val_status = preprocess_new_val(experiment)
        eval_mae = evaluate_production_model(preproc_train_status, preproc_val_status)
        train_mae = re_train(preproc_train_status, preproc_val_status)
        notify(eval_mae, train_mae)
    return flow
