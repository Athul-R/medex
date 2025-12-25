
MAIN_DIR = "/Users/athulr/Documents/Medex/PythonProject/classifier"
EHR_DATA_PATH = f"{MAIN_DIR}/data/synthetic_ehr_notes_gemini.csv"
TRAIN_PATH = f"{MAIN_DIR}/data/train.csv"
VAL_PATH = f"{MAIN_DIR}/data/val.csv"
TEST_PATH = f"{MAIN_DIR}/data/test.csv"

num_cols = ["age", "bp_systolic"]
cat_cols = ["sex", "diagnosis", "medications", "smoker"]
