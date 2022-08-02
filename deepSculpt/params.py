from tensorflow.random import normal

LOCALLY = False

CREATE_DATA = True

N_SAMPLES = 5000

VOID_DIM = 24

BUFFER_SIZE = 5000

BATCH_SIZE = 32

EPOCHS = 5

NOISE_DIM = 1024

SCULPTS_GEN = 1

SEED = normal([SCULPTS_GEN, NOISE_DIM])

BUCKET_NAME = "deepsculpt"

BUCKET_TRAIN_DATA_PATH = "data"

MODEL_BASE_PATH = ""

"""        void_dim,
        n_edge_elements,
        n_plane_elements,
        n_volume_elements,
        color_edges,
        color_planes,
        color_volumes,
        element_edge_min,
        element_edge_max,
        element_grid_min,
        element_grid_max,
        element_plane_min,
        element_plane_max,
        element_volume_min,
        element_volume_max,
        step,
        verbose,"""
