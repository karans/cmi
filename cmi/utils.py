from sklearn.preprocessing import normalize
from collections import defaultdict


def discretize_features(df, cols, n_buckets=5, norm='max'):
    # create n_buckets distinct values for each column
    # todo: update the bucketing strategy to follow the CDF instead of a fixed range
    for col_name in cols:
        col = df[col_name]
        bucket_size = int((max(col) - min(col))/n_buckets)
        df[col_name] = col.apply(lambda val: int(val/bucket_size))
    #normalize along the features
    normalized_features = normalize(df.values, norm=norm, axis=0).T
    for col_name, norm_feat in zip(df.columns, normalized_features):
        df[col_name] = norm_feat


def feature_range_map(df, df_feat, cols):
    col_bucket_range_map = {}
    for col in cols:
        orig_col = df[col]
        feat_col = df_feat[col]
        feat_col_values = defaultdict(list)
        for feat_val, orig_val in zip(feat_col, orig_col):
            feat_col_values[feat_val].append(orig_val)
        feat_col_values = {feat_val: (min(orig_vals), max(orig_vals)) for feat_val, orig_vals in feat_col_values.items()}
        col_bucket_range_map[col] = feat_col_values
    return col_bucket_range_map


def class_func_exists(cls, func):
    return callable(getattr(cls, func, None))