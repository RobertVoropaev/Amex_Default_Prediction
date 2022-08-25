import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from .features_data import *
    
CAT_FEATURES_NA_VALUES = -10
DATA_TYPE = np.float32
    
def get_cat_features(train):
    cat_features = []
    for cat_col in cat_features_origin:
        cat_features.extend([col for col in train.columns if cat_col in col and 
                             ("median" in col or "last" in col or "first" in col)])
    return cat_features


def plot_features_hist(train, feature_list, quantile_alpha: float = 0.05):
    for name in sorted(feature_list, key=lambda x: (x.split("_")[0], int(x.split("_")[1]))):
        plt.figure(figsize=(14, 7))

        feature = train[name]
        min_x = feature.quantile(quantile_alpha)
        max_x = feature.quantile(1 - quantile_alpha)

        sns.histplot(train[feature.between(min_x, max_x)], x=name, hue="target")

        plt.xlim(min_x, max_x)
        plt.title(name + f", fill: {(~feature.isna()).sum()}, na: {feature.isna().sum()}")

        plt.show()
     
    
class AmexFeatures:    
    def __init__(self):
        self.num_origin_features, self.cat_origin_features = self._get_origin_features_type()
        self.origin_groups = self._get_origin_features_group()
    
    def _get_origin_features_type(self):
        sample = pd.read_csv("../input/train_data.csv", nrows=10)

        num_features = list(
            set(sample.columns) - set(cat_features_origin) - set(["customer_ID", "S_2"])
        )

        return num_features, cat_features_origin
    
    def _get_origin_features_group(self):
        feature_groups = origin_feature_groups
        
        feature_groups["other"] = list(
            set(self.num_origin_features) - 
            set(feature_groups["uniform"]) - set(feature_groups["binary"]) - 
            set(feature_groups["unknown"]) - set(feature_groups["discrete"])
        )
        
        feature_groups["category"] = self.cat_origin_features
        
        return feature_groups
    
    def get_origin_feature_group(self, name):
        feature2group = {feature: group for group, feature in self.origin_groups}
        return feature2group[name]
    
    def is_categorical(self, feature_name):
        if feature_name in ["customer_ID", "target", "S_2"]:
            return False

        base_name = "_".join(feature_name.split("_")[:2])
        agg = "_".join(feature_name.split("_")[2:-1])

        return (base_name in self.cat_origin_features) and (agg not in ["count", "nunique"])
    
    def is_numeric(self, feature_name):
        if feature_name in ["customer_ID", "target", "S_2"]:
            return False

        return not self.is_categorical(feature_name)

    def get_categorical(self, train):
        cols = []
        for col in train.columns:
            if self.is_categorical(col):
                cols.append(col)
        return cols
    
    def get_numeric(self, train):
        cols = []
        for col in train.columns:
            if self.is_numeric(col):
                cols.append(col)
        return cols

    
def transform_original_cat_features(train, cat_features, 
                                    cat_features_na_values=CAT_FEATURES_NA_VALUES):
    train['D_63'] = train['D_63'].map({"CO": 0, "CR": 1, "CL": 2, "XZ": 3, "XM": 4, "XL": 5})
    train['D_64'] = train['D_64'].map({"O": 0, "U": 1, "R": 2, "-1": 3})
    
    train[cat_features] = train[cat_features].fillna(cat_features_na_values).astype(DATA_TYPE)
    return train