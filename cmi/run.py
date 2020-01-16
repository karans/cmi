import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import discretize_features
from model_interpretation import FeatureImportance
from lattice import LatticeCMI

### setup dataframe ###
pd.options.mode.chained_assignment = None  # suppress set with copy warning
df = pd.read_csv('../data/heloc_dataset_v1.csv')
df_feat = df[df.columns[:5]]  # use a subset of the table for testing

label_col = df_feat.columns[0]
attribute_cols = df_feat.columns[1:]

df_feat['RiskPerformance'] = df_feat.RiskPerformance.astype('category').cat.codes # bad -> 0, good -> 1
discretize_features(df_feat, attribute_cols) # output column is already normalized.

### train model ###
y = df_feat.RiskPerformance.values
X = df_feat.drop('RiskPerformance', axis=1).values

lr = LogisticRegression(solver='lbfgs')
lr.fit(X, y)

### run cmi ###
feature_importance = FeatureImportance(lr, label_col, attribute_cols)
cmi = LatticeCMI(df_feat, feature_importance, min_similarity=0.01)
contexts = cmi.generate_return_set()

print(contexts)
