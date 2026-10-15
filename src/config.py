"""Central settings. Change values here, not all over the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "chest_xray"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = ROOT / "models" / "pneumonia_mobilenetv2.keras"
OUTPUT_DIR = ROOT / "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
FINE_TUNE_EPOCHS = 4
SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)