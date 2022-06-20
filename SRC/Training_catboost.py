import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import config


def main():
    df = pd.read_csv(config.PREPROCESSED_TRAIN_FILE)
    df_test = pd.read_csv(config.TEST)
    print(df.head())
    catboost(df)
    #xgboost(df)
    return df

def catboost(df):
    X = df.drop(["cost_category"], axis=1)
    y = df["cost_category"]
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=12)
    model = CatBoostClassifier()
    categorical_features_indices = np.where(X_validate.dtypes !=np.float)[0]
    model  = CatBoostClassifier(iterations=50, depth=3, learning_rate=0.3, early_stopping_rounds=10, loss_function='MultiClass')
    model.fit(X_train, y_train, cat_features=categorical_features_indices,  eval_set=(X_validate, y_validate),plot=True)
    predicted_y_val = model.predict_proba((X_validate))
    print(predicted_y_val)
    return df

#def xgboost(df):
    X = df.drop(["cost_category"], axis=1)
    y = df["cost_category"]
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=None)
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    predicted_y_val = model.predict_proba(X_validate)
    print(predicted_y_val)
    return df

#def LGBM(df):
    X = df.drop(["cost_category"], axis=1)
    y = df["cost_category"]

    return df 


if __name__ == "__main__":
    main()