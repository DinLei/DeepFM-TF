
# set the path-to-files
TRAIN_FILE = "./data/adult.data"
TEST_FILE = "./data/adult.test"

SUB_DIR = "./output"

NUM_CK_POINTS = 10

# 每2个step就保存一次checkpoint
CHECKPOINT_EVERY = 5


NUM_SPLITS = 3
RANDOM_SEED = 2019

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex",
    "native_country"
]

NUMERIC_COLS = [
    # # binary
    # numeric
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"

    # feature engineering
]

IGNORE_COLS = [
    "id", "target"
]


FEATURE_COLS = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "target"
]
