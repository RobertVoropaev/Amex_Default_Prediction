import pandas as pd
from tqdm.auto import tqdm

import gc

from .features_tools import *

def get_num_agg_for_month(train, num_features, months_ago, agg_func_list):
    date_start = (
        pd.to_datetime(train["S_2"].max()) - pd.Timedelta(days=months_ago*30)
    ).strftime("%Y-%m-%d")
    train = train[train["S_2"] >= date_start]
    
    train_num_agg = (
        train.groupby("customer_ID")[num_features].agg(agg_func_list)
            .astype(DATA_TYPE)
    )
    
    train_num_agg.columns = ['_'.join(x) + f"_{months_ago}m" for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace = True)
    
    return train_num_agg

def get_num_agg(train, num_features, num_month_agg, agg_func_list):
    train_num_agg = None
    for month_ago in tqdm(num_month_agg):
        train_num_agg_month = get_num_agg_for_month(train, num_features, 
                                                    months_ago=month_ago, 
                                                    agg_func_list=agg_func_list)

        if train_num_agg is not None:
            train_num_agg = train_num_agg.merge(train_num_agg_month, on="customer_ID", how="outer") 
        else:
            train_num_agg = train_num_agg_month

    return train_num_agg

def add_num_last_ratio(train_num_agg, num_features, 
                       num_month_agg, agg_func_list, 
                       eps=1e-6):
    for col in tqdm(num_features):
        for month in num_month_agg:
            for agg in agg_func_list:
                if agg != "last":
                    train_num_agg[f"{col}_last_{agg}_ratio_{month}m"] = (
                        train_num_agg[f"{col}_last_{month}m"] / (train_num_agg[f"{col}_{agg}_{month}m"] + eps)
                    ).astype(DATA_TYPE)

    return train_num_agg


def add_num_last_diff(train_num_agg, num_features, 
                     num_month_agg, agg_func_list):
    for col in tqdm(num_features):
        for month in num_month_agg:
            for agg in agg_func_list:
                if agg != "last":
                    train_num_agg[f"{col}_last_{agg}_diff_{month}m"] = (
                        train_num_agg[f"{col}_last_{month}m"] - (train_num_agg[f"{col}_{agg}_{month}m"])
                    ).astype(DATA_TYPE)

    return train_num_agg


########################################################################

def get_cat_agg_for_month(train, cat_features, months_ago, agg_func_list):
    date_start = (
        pd.to_datetime(train["S_2"].max()) - pd.Timedelta(days=months_ago*30)
    ).strftime("%Y-%m-%d")
    train = train[train["S_2"] >= date_start]
    
    train_cat_agg = (
        train.groupby("customer_ID")[cat_features].agg(agg_func_list)
            .astype(DATA_TYPE)
    )
    
    train_cat_agg.columns = ['_'.join(x) + f"_{months_ago}m" for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace = True)

    return train_cat_agg

def get_cat_agg(train, cat_features, 
                agg_func_list, cat_month_ago, 
                cat_features_na_values=CAT_FEATURES_NA_VALUES):
    train_cat_agg = None
    for month_ago in tqdm(cat_month_ago):
        train_cat_agg_month = get_cat_agg_for_month(train, cat_features, 
                                                    months_ago=month_ago, 
                                                    agg_func_list=agg_func_list)

        if train_cat_agg is not None:
            train_cat_agg = train_cat_agg.merge(train_cat_agg_month, on="customer_ID", how="outer") 
        else:
            train_cat_agg = train_cat_agg_month

    train_cat_agg = train_cat_agg.fillna(cat_features_na_values)
            
    return train_cat_agg

########################################################################

def preprocessing(train, 
                  num_features, cat_features, 
                  months_agg_list, 
                  num_agg_func_list, cat_agg_func_list,
                  is_test: bool = False):
    
    train_num_agg = get_num_agg(train, num_features, 
                                num_month_agg = months_agg_list, 
                                agg_func_list = num_agg_func_list)
    
    train_num_agg = add_num_last_ratio(train_num_agg, num_features,
                                       num_month_agg = months_agg_list, 
                                       agg_func_list = num_agg_func_list)
    
    train_num_agg = add_num_last_diff(train_num_agg, num_features, 
                                      num_month_agg = months_agg_list, 
                                      agg_func_list = num_agg_func_list)
    
    
    if cat_features != []:
        train_cat_agg = get_cat_agg(train, cat_features, 
                                    cat_month_ago = months_agg_list,
                                    agg_func_list = cat_agg_func_list)
        
        train_agg = (
            train_num_agg
                .merge(train_cat_agg, how = 'outer', on = 'customer_ID')
        )
        del train_cat_agg
    else:
        train_agg = train_num_agg
    
    del train_num_agg, train
    gc.collect()
    
    if not is_test:
        train_labels = pd.read_csv("../input/train_labels.csv")
        train_labels["target"] = train_labels["target"].astype(DATA_TYPE)
        train_agg = train_agg.merge(train_labels, how = 'inner', on = 'customer_ID')
    
    return train_agg