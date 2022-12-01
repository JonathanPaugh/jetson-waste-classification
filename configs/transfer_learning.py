from models.tensorflow_hub import InceptionV3

# NOTE: unfreeze breakpoints and chosen model are generally codependent
TRANSFER_LEARNING_BASE_MODEL = InceptionV3
FINE_TUNING_UNFREEZE_BREAKPOINTS = None

FINE_TUNING_ENABLED = True
FINE_TUNING_NUM_EPOCHS_PER_BREAKPOINT = 50
FINE_TUNING_LEARNING_RATE = 1e-5         # defaults to 1e-4
FINE_TUNING_LEARNING_RATE_DECAY = 1 / 2  # multiply base rate by this value once per breakpoint
