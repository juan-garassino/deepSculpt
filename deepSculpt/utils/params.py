from tensorflow.random import normal
import numpy as np
import os

## TRAINING PARAMS

BUFFER_SIZE = int(int(os.environ.get("TRAIN_SIZE")) / 10)

MINIBATCHES = np.arange(
    0,
    int(os.environ.get("TRAIN_SIZE")),
    int(os.environ.get("TRAIN_SIZE")) / (int(os.environ.get("TRAIN_SIZE")) / 10),
)

SEED = normal([int(os.environ.get("SCULPTS_GEN")), int(os.environ.get("NOISE_DIM"))])

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

ELEMENT_EDGE_MIN, ELEMENT_EDGE_MAX = int(int(os.environ.get("VOID_DIM")) * 0.8), int(
    int(os.environ.get("VOID_DIM")) * 0.9
)

ELEMENT_GRID_MIN, ELEMENT_GRID_MAX = int(int(os.environ.get("VOID_DIM")) * 0.8), int(
    int(os.environ.get("VOID_DIM")) * 0.95
)

ELEMENT_PLANE_MIN, ELEMENT_PLANE_MAX = int(int(os.environ.get("VOID_DIM")) * 0.4), int(
    int(os.environ.get("VOID_DIM")) * 0.8
)

ELEMENT_VOLUME_MIN, ELEMENT_VOLUME_MAX = int(int(os.environ.get("VOID_DIM")) * 0.2), int(
    int(os.environ.get("VOID_DIM")) * 0.5
)

STEP = 1

VERBOSE = False
