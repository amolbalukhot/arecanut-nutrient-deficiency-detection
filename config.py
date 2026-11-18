"""Central configuration. Deployment-specific values come from environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CLASS_NAMES = [
    "Healthy",
    "Nitrogen_Deficiency",
    "Potassium_Deficiency",
    "Magnesium_Deficiency",
]
VERIFICATION_CLASS_NAMES = ["Arecanut", "Non_Arecanut"]  # index 0 -> Arecanut, 1 -> Non_Arecanut

VERIFICATION_DATASET_DIR = Path(os.getenv("VERIFICATION_DATASET_DIR", BASE_DIR / "verification_dataset"))
NUTRIENT_DATASET_DIR = Path(os.getenv("NUTRIENT_DATASET_DIR", BASE_DIR / "dataset"))

SAVED_MODELS_DIR = BASE_DIR / "saved_models"
VERIFICATION_MODEL_PATH = SAVED_MODELS_DIR / "verification_model.keras"
MOBILENETV2_MODEL_PATH = SAVED_MODELS_DIR / "mobilenetv2_model.keras"

VERIFICATION_IMAGE_SIZE = (128, 128)   # binary CNN input; pixels scaled to [0,1] by a Rescaling layer inside the model
NUTRIENT_IMAGE_SIZE = (224, 224)       # MobileNetV2 input; preprocess_input applied OUTSIDE the model (once)

# Verification output: sigmoid = P(Non_Arecanut) (class index 1, alphabetical-free, fixed order above).
VERIFICATION_THRESHOLD = float(os.getenv("VERIFICATION_THRESHOLD", "0.60"))

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "16")) * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "database" / "predictions.db"))
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
HEATMAP_DIR = BASE_DIR / "static" / "heatmaps"
REPORT_DIR = BASE_DIR / "static" / "reports"
SPLIT_DIR = BASE_DIR / "data_splits"
SPLIT_REPORT_DIR = SPLIT_DIR / "reports"
TRAINING_RESULTS_DIR = BASE_DIR / "training_results"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
