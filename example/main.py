import os
import sys

# import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from utils import config
from utils.data_reader import FeatureDictionary, DataParser
from utils.metrics import *

sys.path.append("..")
from models.DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data():
    print("Reading data...")
    dfTrain = pd.read_csv(config.TRAIN_FILE, header=None)
    dfTest = pd.read_csv(config.TEST_FILE, header=None)

    dfTrain.columns = config.FEATURE_COLS
    dfTest.columns = config.FEATURE_COLS

    le = LabelEncoder()
    le.fit([" >50K", " <=50K"])

    def preprocess(df):
        if "target" in df.columns:
            df["target"] = le.transform(df["target"])
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (c not in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values

    if "target" in dfTest.columns:
        y_test = dfTest["target"].values
    else:
        y_test = None
    if "id" in dfTest.columns:
        ids_test = dfTest["id"].values
    else:
        ids_test = list(dfTest.index)
    # cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, y_test, ids_test


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest, has_label=True)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    eval_metric = dfm_params["eval_metric"]

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)

    _get = lambda x, l: [x[i] for i in l]
    eval_metric_results_cv = np.zeros(len(folds), dtype=float)
    eval_metric_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    eval_metric_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

    dfm = DeepFM(**dfm_params)

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)

        eval_metric_results_cv[i] = eval_metric(y_valid_, y_train_meta[valid_idx])
        eval_metric_results_epoch_train[i] = dfm.train_result
        eval_metric_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, eval_metric_results_cv.mean(), eval_metric_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, eval_metric_results_cv.mean(), eval_metric_results_cv.std())
    _make_submission(ids_test, y_test_meta, y_test, filename)

    _plot_fig(eval_metric_results_epoch_train, eval_metric_results_epoch_valid, clf_str)

    _export_embedding(dfm.sess, fd.feat_dict, dfm.weights["feature_embeddings"],
                      config.CATEGORICAL_COLS, config.SUB_DIR)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, y_truth=None, filename="submission.csv"):
    sub_path = os.path.abspath(os.path.join(config.SUB_DIR, "metricfiles"))
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    data = {"id": ids, "prediction": y_pred.flatten()}
    if y_truth is not None:
        data["label"] = y_truth
    pd.DataFrame(data).to_csv(
        os.path.join(sub_path, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i + 1))
        legends.append("valid-%d" % (i + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png" % model_name)
    plt.close()


def _export_embedding(sess, feat_dict, feat_embeddings,
                      category_cols=[], embed_dir="."):
    for col in category_cols:
        print("saving {}'s embedding matrix... ".format(col))
        sub_path = os.path.join(embed_dir, "embeddings")
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        sub_feat_dict = feat_dict[col]
        names = sub_feat_dict.keys()
        indexes = list(sub_feat_dict.values())
        embed_val = sess.run(tf.nn.embedding_lookup(feat_embeddings, indexes))
        with open(os.path.join(sub_path, col+".txt"), "w") as fin:
            for i, n in enumerate(names):
                embed = embed_val[i, :]
                fin.write('%s\t%s\n' % (n, ' '.join(map(str, embed))))


if __name__ == "__main__":

    # load data
    dfTrain, dfTest, X_train, y_train, X_test, y_test, ids_test = _load_data()

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))

    # ------------------ DeepFM Model ------------------
    # params
    dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 8,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "eval_metric": precision,
        "random_seed": config.RANDOM_SEED,
        "num_checkpoints": config.NUM_CK_POINTS,
        "model_dir": config.SUB_DIR,
        "checkpoint_every": config.CHECKPOINT_EVERY
    }

    y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

    if False:
        # ------------------ FM Model ------------------
        fm_params = dfm_params.copy()
        fm_params["use_deep"] = False
        y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)

        # ------------------ DNN Model ------------------
        dnn_params = dfm_params.copy()
        dnn_params["use_fm"] = False
        y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)
