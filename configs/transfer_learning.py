from models.tensorflow_hub import InceptionV3

# NOTE: unfreeze breakpoints and chosen model are generally codependent
TRANSFER_LEARNING_MODEL = InceptionV3
FINE_TUNING_UNFREEZE_BREAKPOINTS = ('mixed8',)  # 'mixed6', 'mixed4')

FINE_TUNING_ENABLED = True
FINE_TUNING_NUM_EPOCHS = 1
FINE_TUNING_LEARNING_RATE = 1e-5         # defaults to 1e-4
FINE_TUNING_LEARNING_RATE_DECAY = 1 / 2  # multiply by this value per fine tuning round
