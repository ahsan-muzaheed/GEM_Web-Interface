from torch.utils.tensorboard import SummaryWriter
import enum
import os

# 3 different model training/eval phases used in train.py
class LoopPhase(enum.Enum):
    TRAIN = (0,)
    VAL = (1,)
    TEST = 2


writer = (
    SummaryWriter()
)  # (tensorboard) writer will output to ./runs/ directory by default


# Global vars used for early stopping. After some number of epochs (as defined by the patience_period var) without any
# improvement on the validation dataset (measured via accuracy metric), we'll break out from the training loop.
BEST_VAL_ACC = 0
BEST_VAL_LOSS = 0
PATIENCE_CNT = 0

BINARIES_PATH = os.path.join(os.getcwd(), "python", "binarypath", "binaries")
CHECKPOINTS_PATH = os.path.join(os.getcwd(), "python", "binarypath", "checkpoints")

# Make sure these exist as the rest of the code assumes it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
