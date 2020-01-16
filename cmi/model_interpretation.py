from utils import class_func_exists
import numpy as np
from sklearn.metrics import roc_auc_score


class ModelInterpretation:
    def __init__(self, model, label_col, attribute_cols):
        if not class_func_exists(model, 'predict_proba'):
            raise RuntimeError(
                f'Model {type(model)} does not have required predict_proba function needed to calculate AUC')
        self.model = model
        self.label_col = label_col
        self.attribute_cols = attribute_cols

    def context_to_features(self, context):
        return context.df[self.attribute_cols].values, context.df[self.label_col].values

    def on(self, context):
        if class_func_exists(self, 'get_weights_decomp'):
            return self.get_weights_decomp(context)
        else:
            return self.get_weights(context)

    def get_weights(self, context):
        raise NotImplementedError


class FeatureImportance(ModelInterpretation):
    def __init__(self, model, label_col, attribute_cols):
        super(FeatureImportance, self).__init__(model, label_col, attribute_cols)

    def error(self, X, y):
        y_pred = self.model.predict_proba(X)
        class_idx = np.argwhere(self.model.classes_ == 1)[0]
        class_prob = y_pred[:, class_idx]  # get the probabilities of the class label 1
        error = 1 - roc_auc_score(y, class_prob)
        return error

    def get_weights(self, context):
        feature_importances = []
        X, y = self.context_to_features(context)
        global_error = self.error(X, y)

        # destroy each feature by shuffling and record the magnitude increase in error
        for col in range(len(X[0])):
            X_perm = X.copy()
            X_perm[:, col] = X_perm[np.random.permutation(len(X)), col]
            feature_error = self.error(X_perm, y)
            feature_importance = feature_error / global_error
            feature_importances.append(feature_importance)

        # give normalized feature weights
        return np.array(feature_importances) / np.sum(feature_importances)
