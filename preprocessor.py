from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from config import num_cols, cat_cols, TRAIN_PATH, VAL_PATH, TEST_PATH
import pandas as pd



def column_processor():
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )   
    return preprocess

def get_train_test_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df


def preprocess_data(df):
    X, y = df.drop("readmitted_30d", axis=1), df["readmitted_30d"]
    return X, y
