from pathlib import Path

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
RANDOM_SEED = 42
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

RESULTS_DIR = Path("results")
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUTS_DIR = Path("outputs")
GRADCAM_OUTPUTS_DIR = Path("gradcam_outputs")

STANFORD_FRAMES_PER_PATIENT = 5
STANFORD_TIRADS_COLUMNS = [
    "ti_rads_composition",
    "ti_rads_echogenicity",
    "ti_rads_shape",
    "ti_rads_margin",
    "ti_rads_echogenicfoci",
    "ti_rads_level",
]

THYROIDXL_B1_NUMERIC_COLUMNS = ["nodule_width", "nodule_height", "nodule_depth"]
THYROIDXL_B1_CATEGORICAL_COLUMNS = ["nodule_position"]
THYROIDXL_B2_NUMERIC_COLUMNS = []
THYROIDXL_B2_CATEGORICAL_COLUMNS = ["final_tirads"]
