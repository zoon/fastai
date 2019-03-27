from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from scipy.cluster import hierarchy as hc
from treeinterpreter import treeinterpreter as ti

from sklearn import metrics
import feather
import pdpbox.pdp as pdp
from plotnine import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
from contextlib import contextmanager

plt.rcParams["figure.figsize"] = (10, 6)
set_plot_sizes(12, 14, 16)
np.set_printoptions(precision=5)
pd.set_option("display.precision", 5)
%precision 5

# constants #################################################################
PATH = 'data/bulldozers/'
DF_RAW_PATH = 'tmp/bulldozers-raw'
N_VALID = 12_000


# defines: ##################################################################
@contextmanager
def rf_samples(n='all', verbose=None):
    if isinstance(n, int) and n > 0:
        set_rf_samples(n)
        if verbose:
            print('set_rf_samples', n)
    try:
        yield
    finally:
        if isinstance(n, int) and n > 0:
            reset_rf_samples()
            if verbose:
                print('reset_rf_samples')


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns",
                           1000):
        display(df)


def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()


def rmse(x, y):
    return math.sqrt(((x - y)**2).mean())


def do_print_score(m, X_train, y_train, X_valid, y_valid):
    res = [
        rmse(m.predict(X_train), y_train),
        rmse(m.predict(X_valid), y_valid),
        m.score(X_train, y_train),
        m.score(X_valid, y_valid),
    ]
    if hasattr(m, "oob_score_"):
        res.append(m.oob_score_)
    print(np.array(res))


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def do_proc_df(
        df_raw,
        n_valid,
        y_fld=None,
        skip_flds=None,
        ignore_flds=None,
        do_scale=False,
        na_dict=None,
        preproc_fn=None,
        max_n_cat=None,
        subset=None,
        mapper=None,
        to_keep=None,
):
    df_trn, y_trn, nas = proc_df(
        df=df_raw,
        y_fld=y_fld,
        skip_flds=skip_flds,
        ignore_flds=ignore_flds,
        do_scale=do_scale,
        na_dict=na_dict,
        preproc_fn=preproc_fn,
        max_n_cat=max_n_cat,
        subset=subset,
        mapper=mapper)
    if not (to_keep is None):
        df_trn = df_trn[to_keep]
    n_trn = len(df_trn) - n_valid
    X_train, X_valid = split_vals(df_trn, n_trn)
    y_train, y_valid = split_vals(y_trn, n_trn)
    raw_train, raw_valid = split_vals(df_raw, n_trn)
    return AttrDict(locals())


# variant with limited max_depth
def do_draw_tree(tree,
                 df,
                 max_depth=2,
                 size=10,
                 ratio=0.6,
                 precision=3,
                 rotate=True):
    s = export_graphviz(
        tree,
        out_file=None,
        feature_names=df.columns,
        filled=True,
        special_characters=True,
        rotate=rotate,
        max_depth=max_depth,
        precision=precision,
    )
    display(
        graphviz.Source(
            re.sub("Tree {", f"Tree {{ size={size}; ratio={ratio}", s)))