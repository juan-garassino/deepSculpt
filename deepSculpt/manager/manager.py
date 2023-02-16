from matplotlib import pyplot as plt
from colorama import Fore, Style
from google.cloud import storage
from tensorflow.data import Dataset
import os
import numpy as np
import errno
import mlflow
from mlflow.tracking import MlflowClient
import imageio


class Manager:  # make manager work with and with out epochs
    def __init__(
        self, model_name=None, data_name=None, path_colors=None, path_volumes=None
    ):
        self.model_name = model_name

        self.path_volumes = path_volumes

        self.path_colors = path_colors

        self.data_name = data_name

        self.comment = "{}_{}".format(model_name, data_name)

        self.data_subdir = "{}/{}".format(model_name, data_name)

    def upload_snapshot_to_gcp(snapshot_name):

        STORAGE_FILENAME = snapshot_name

        storage_location = f"results/{STORAGE_FILENAME}"

        bucket = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

        blob = bucket.blob(storage_location)

        blob.upload_from_filename(STORAGE_FILENAME)

        print(
            "\n 🔼 "
            + Fore.BLUE
            + f"Just uploaded a snapshot to gcp {STORAGE_FILENAME} @ {storage_location}"
            + Style.RESET_ALL
        )

    def load_locally(self):

        raw_data = np.load(self.path_volumes, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        color_raw_data = np.load(self.path_colors, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        print(
            "\n🔼 "
            + Fore.BLUE
            + f"Just loaded 'volume_data' shaped {raw_data.shape} and 'material_data' shaped{color_raw_data.shape}"
            + Style.RESET_ALL
        )

        return (raw_data, color_raw_data)

    def load_from_gcp(self):

        files = [self.path_volumes, self.path_colors]

        client = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

        for file in files:

            blob = client.blob(os.environ.get("BUCKET_TRAIN_DATA_PATH") + "/" + file)

            blob.download_to_filename(file)

        raw_data = np.load(self.path_volumes, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        color_raw_data = np.load(self.path_colors, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        print(
            "\n 🔼 "
            + Fore.BLUE
            + f"Just loaded 'volume_data' shaped {raw_data.shape} and 'material_data' shaped{color_raw_data.shape}"
            + Style.RESET_ALL
        )

        return (raw_data, color_raw_data)

    def save_mlflow_model(metrics=None, params=None, model=None):
        # retrieve mlflow env params
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        # configure mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment)

        with mlflow.start_run():

            # STEP 1: push parameters to mlflow
            if params is not None:
                mlflow.log_params(params)

            # STEP 2: push metrics to mlflow
            if metrics is not None:
                mlflow.log_metrics(metrics)

            # STEP 3: push model to mlflow
            if model is not None:

                mlflow.keras.log_model(
                    keras_model=model,
                    artifact_path="model",
                    keras_module="tensorflow.keras",
                    registered_model_name=mlflow_model_name,
                )

        print("\n ✅ Data saved in mlflow")

        return None

    def load_mlflow_model():
        stage = "Production"

        print(
            Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL
        )

        # load model from mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")

        try:
            model = mlflow.keras.load_model(model_uri=model_uri)
            print("\n✅ model loaded from mlflow")
        except:
            print(f"\n❌ no model in stage {stage} on mlflow")
            return None

        return model

    def get_model_version(stage="Production"):
        """
        Retrieve the version number of the latest model in the given stage
        - stages: "None", "Production", "Staging", "Archived"
        """

        import mlflow
        from mlflow.tracking import MlflowClient

        if os.environ.get("MODEL_TARGET") == "mlflow":

            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

            mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

            client = MlflowClient()

            try:
                version = client.get_latest_versions(
                    name=mlflow_model_name, stages=[stage]
                )
            except:
                return None

            # check whether a version of the model exists in the given stage
            if not version:
                return None

            return int(version[0].version)

        # model version not handled

        return None

    def clean_data(df):
        pass

    def holdout(df):
        pass

    @staticmethod
    def return_axis(void: np.ndarray, color_void: np.ndarray) -> tuple:
        """
        Selects a random plane from a 3D numpy array along a random axis.

        Args:
            void (np.ndarray): The 3D numpy array to select a plane from.
            color_void (np.ndarray): The 3D numpy array that holds the color information.

        Returns:
            tuple: A tuple containing:
                - working_plane (np.ndarray): The randomly selected plane.
                - color_parameters (np.ndarray): The color information of the selected plane.
                - section (int): The index of the selected plane.
        """
        section = np.random.randint(low=0, high=void.shape[0])
        axis_selection = np.random.randint(low=0, high=3)

        if axis_selection == 0:
            working_plane = void[section, :, :]
            color_parameters = color_void[section, :, :]
        elif axis_selection == 1:
            working_plane = void[:, section, :]
            color_parameters = color_void[:, section, :]
        elif axis_selection == 2:
            working_plane = void[:, :, section]
            color_parameters = color_void[:, :, section]
        else:
            print("Error: axis_selection value out of range.")

        return working_plane, color_parameters, section

    @staticmethod
    def verbose(*args, **kwargs):
        """Prints input arguments and keyword arguments in a nice, formatted way."""
        print("=" * 50)
        print("Verbose output:")
        print("-" * 50)

        if args:
            print("Arguments:")
            for arg in args:
                print(f"  {arg}")

        if kwargs:
            print("Keyword arguments:")
            for key, value in kwargs.items():
                print(f"  {key}: {value}")

        print("=" * 50)

    @staticmethod
    def create_animation(frames_path, output_name="animation", fps=30):
        # Get a list of all image files in the directory
        image_files = sorted(
            [
                f
                for f in os.listdir(frames_path)
                if f.endswith(".png") or f.endswith(".jpg")
            ]
        )

        # Load the image files into an array of image arrays
        images = [imageio.imread(os.path.join(frames_path, f)) for f in image_files]

        # Create the animation and save it as a GIF file
        animation_path = os.path.join(frames_path, f"{output_name}.gif")

        imageio.mimsave(animation_path, images, fps=fps)

    @staticmethod
    def make_directory(directory):
        try:
            os.makedirs(directory)

            print(
                "\n ⏹ "
                + Fore.GREEN
                + f" This directory has been created {directory}"
                + Style.RESET_ALL
            )

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    pass
