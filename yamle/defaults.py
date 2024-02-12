TINY_EPSILON = 1e-8
POSITIVE_INFINITY = 2147483647
NEGATIVE_INFINITY = -POSITIVE_INFINITY
MEMBERS_DIM = 1
TARGET_KEY = "y"
INPUT_KEY = "x"
TARGET_PER_MEMBER_KEY = "y_permember"
PREDICTION_KEY = "y_hat"
MEAN_PREDICTION_KEY = "y_hat_mean"
PREDICTION_PER_MEMBER_KEY = "y_hat_permember"
AVERAGE_WEIGHTS_KEY = "average_weights"

LOSS_KEY = "loss"
LOSS_REGULARIZER_KEY = f"{LOSS_KEY}_regularizer"
LOSS_KL_KEY = f"{LOSS_KEY}_kl"

REGRESSION_KEY = "regression"
CLASSIFICATION_KEY = "classification"
TEXT_CLASSIFICATION_KEY = "text_classification"
SEGMENTATION_KEY = "segmentation"
RECONSTRUCTION_KEY = "reconstruction"
DEPTH_ESTIMATION_KEY = "depth_estimation"
PRE_TRAINING_KEY = "pre_training"
SUPPORTED_TASKS = [
    REGRESSION_KEY,
    CLASSIFICATION_KEY,
    TEXT_CLASSIFICATION_KEY,
    SEGMENTATION_KEY,
    DEPTH_ESTIMATION_KEY,
    PRE_TRAINING_KEY,
    RECONSTRUCTION_KEY,
]


# Define arguments that can be different when considering averaging the resutls
ARGS_CAN_BE_DIFFERENT = [
    "save_path",
    "seed",
    "trainer_accelerator",
    "trainer_mode",
    "st_checkpoint_dir",
    "trainer_devices",
    "load_path",
    "label",
    "datamodule_pin_memory",
    "trainer_no_validation_saving",
]

TRAIN_KEY = "train"
VALIDATION_KEY = "validation"
TEST_KEY = "test"
ALL_DATASETS_KEY = "all"
CALIBRATION_KEY = "calibration"

MIN_TENDENCY = "min"
MAX_TENDENCY = "max"

TRAIN_DATA_SPLIT_KEY = "split_"
DISABLED_REGULARIZER_KEY = "_no_regularizer"

# This is used to set the optimzier id if there are multiple optimizers
# for which the parameters shoud be split
OPTIMIZER_ID_KEY = "_optimizer_id"

FROZEN_MASK_KEY = "_frozen_mask"
FROZEN_DATA_KEY = "_frozen_data"

DISABLED_OPTIMIZATION_KEY = "_disabled_optimization"
DISABLED_DROPOUT_KEY = "_disabled_dropout"
DISABLED_VI_KEY = "_disabled_vi"
DISABLED_PRUNING_KEY = "_disabled_pruning"
FORMER_DATA_PRUNING_KEY = "_former_data"
MASK_PRUNING_KEY = "_pruning_mask"

MODULE_INPUT_SHAPE_KEY = "_input_shape"
MODULE_OUTPUT_SHAPE_KEY = "_output_shape"
MODULE_HARDWARE_PROPERTIES_KEY = "_hardware_properties"
MODULE_FLOPS_KEY = "__flops__"
MODULE_PARAMS_KEY = "__params__"
MODULE_CUMULATIVE_FLOPS_KEY = "__cumulative_flops__"  # These are flops which are accumulated from all the previous layers
MODULE_CUMULATIVE_PARAMS_KEY = "__cumulative_params__"  # These are params which are accumulated from all the previous layers
MODULE_NAME_KEY = "_name"


QUANTIZED_MODEL_KEY = "_quantized_model"
FLOAT_MODEL_KEY = "_float_model"
QUANTIZED_KEY = "_quantized"

FIT_TIME_KEY = "fit_time"
TEST_TIME_KEY = "test_time"
PROFILING_TIME_KEY = "profiling_time"
