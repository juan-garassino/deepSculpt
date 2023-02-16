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

N_EDGE_ELEMENTS = 0

N_PLANE_ELEMENTS = 0

N_VOLUME_ELEMENTS = 7


ELEMENT_EDGE_MIN, ELEMENT_EDGE_MAX = 0.8, 0.9

ELEMENT_PLANE_MIN, ELEMENT_PLANE_MAX = 0.6, 0.85

ELEMENT_VOLUME_MIN, ELEMENT_VOLUME_MAX = 0.3, 0.45

COLORS = dict(edges="dimgrey", planes="snow", volumes=["crimson", "turquoise", "gold"])

STEP = 1

"""[
        "crimson", "turquoise", "gold", "orange", "mediumpurple", "greenyellow",
        "firebrick", "salmon", "coral", "chartreuse", "steelblue", "lavender", "royalblue",
        "indigo", "mediumvioletred"
    ]"""
