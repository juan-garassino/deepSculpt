from tensorflow.random import normal
import numpy as np

## TRAINING PARAMS

LOCALLY = True

COLAB = True

CREATE_DATA = False

N_SAMPLES_CREATE = 100

VOID_DIM = 48

NOISE_DIM = 512

FILE_TO_LOAD_VOLUMES = "raw-data[2022-06-15].npy"

FILE_TO_LOAD_COLORS = "color-raw-data[2022-06-15].npy"


TRAIN_SIZE = 2500

BUFFER_SIZE = int(TRAIN_SIZE / 10)

BATCH_SIZE = 16

MINIBATCHES = np.arange(0, TRAIN_SIZE, TRAIN_SIZE / (TRAIN_SIZE / 10))

EPOCHS = 200


SCULPTS_GEN = 1

SEED = normal([SCULPTS_GEN, NOISE_DIM])

BUCKET_NAME = "deepsculpt"

BUCKET_TRAIN_DATA_PATH = "data"

MODEL_BASE_PATH = ""

MODEL_CHECKPOINT = 20

PICTURE_SNAPSHOT = 1


## ELEMENTS PARAMS


N_EDGE_ELEMENTS = 2

N_PLANE_ELEMENTS = 2

N_VOLUME_ELEMENTS = 2

COLOR_EDGES = "dimgrey"

COLOR_PLANES = "snow"

COLOR_VOLUMES = ["crimson", "turquoise", "gold"]

"""[
        "crimson", "turquoise", "gold", "orange", "mediumpurple", "greenyellow",
        "firebrick", "salmon", "coral", "chartreuse", "steelblue", "lavender", "royalblue",
        "indigo", "mediumvioletred"
    ]"""

ELEMENT_EDGE_MIN, ELEMENT_EDGE_MAX = int(VOID_DIM * 0.8), int(VOID_DIM * 0.9)

ELEMENT_GRID_MIN, ELEMENT_GRID_MAX = int(VOID_DIM * 0.8), int(VOID_DIM * 0.95)

ELEMENT_PLANE_MIN, ELEMENT_PLANE_MAX = int(VOID_DIM * 0.4), int(VOID_DIM * 0.8)

ELEMENT_VOLUME_MIN, ELEMENT_VOLUME_MAX = int(VOID_DIM * 0.2), int(VOID_DIM * 0.5)

STEP = 1

VERBOSE = False
