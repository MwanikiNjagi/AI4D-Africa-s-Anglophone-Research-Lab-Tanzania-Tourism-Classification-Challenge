import config
import pandas as pd
from catboost import CatBoostClassifier
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import re

def main():
    df =  pd.read_csv(config.PREPROCESSED_TEST_FILE)
    df_train = pd.read_csv(config.PREPROCESSED_TRAIN_FILE)
    submission =  pd.read_csv(config.SUBMISSION)
    cat_boost(df, df_train, submission)
    #XGB_boost(df, df_train, submission)
    #LightGB(df, df_train, submission)
    print(df.head())
    print(df_train.head())
    return df

def cat_boost(df, df_train, submission):
    X_train = df_train.drop(["cost_category"], axis=1)
    y_train = df_train["cost_category"]
    X_test =  df.drop(["cost_category"], axis=1)
    y_test =  df["cost_category"]
    model = CatBoostClassifier()
    categorical_features_indices = np.where(X_test.dtypes !=np.float)[0]
    model  = CatBoostClassifier(iterations=50, depth=3, learning_rate=0.3, early_stopping_rounds=10, loss_function='MultiClass')
    model.fit(X_train, y_train, cat_features=categorical_features_indices ,plot=True)
    predicted_y_val = model.predict_proba(pd.DataFrame(X_test))
    #predicted_y_val = np.exp(predicted_y_val)
    print(predicted_y_val)
    #Value Assignement 
    pred_0 = predicted_y_val[:, 0]
    pred_1 = predicted_y_val[:, 1]
    pred_2 = predicted_y_val[:, 2]
    pred_3 = predicted_y_val[:, 3]
    pred_4 = predicted_y_val[:, 4]
    pred_5 = predicted_y_val[:, 5]
    #Storing them in the Submission file
    submission["High Cost"] = pred_0 
    submission["Higher Cost"] = pred_1
    submission["Highest Cost"] = pred_2
    submission["Low Cost"] = pred_3
    submission["Lower Cost"] = pred_4
    submission["Normal Cost"]= pred_5
    #Turning the submission data into a csv
    submission.to_csv("./Input/sub_cat.csv", index = False)
    return df

def XGB_boost(df, df_train, submission):
    X_train = df_train.drop(["cost_category"], axis=1)
    y_train = df_train["cost_category"]
    X_test =  df.drop(["cost_category"], axis=1)
    y_test =  df["cost_category"]
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    predicted_y_val = model.predict_proba(X_test)
    print(np.exp(predicted_y_val))
     #Value Assignement 
    pred_0 = predicted_y_val[:, 0]
    pred_1 = predicted_y_val[:, 1]
    pred_2 = predicted_y_val[:, 2]
    pred_3 = predicted_y_val[:, 3]
    pred_4 = predicted_y_val[:, 4]
    pred_5 = predicted_y_val[:, 5]
    #Storing them in the Submission file
    submission["High Cost"] = pred_0 
    submission["Higher Cost"] = pred_1
    submission["Highest Cost"] = pred_2
    submission["Low Cost"] = pred_3
    submission["Lower Cost"] = pred_4
    submission["Normal Cost"]= pred_5
    #Turning the submission data into a csv
    submission.to_csv("./Input/sub_xgb.csv", index = False)
    return df

def LightGB(df, df_train, submission):
    X_train = df_train.drop(["cost_category"], axis=1)
    y_train = df_train["cost_category"]
    X_test =  df.drop(["cost_category"], axis=1)
    y_test =  df["cost_category"]
    #next two lines remove special characters from feature names
    X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    y_train = y_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    categorical_features_indices = np.where(X_test.dtypes !=np.float)[0]
    model = LGBMClassifier(metric='multi_logloss')
    model.fit(X_train, y_train)
    predicted_y_val = model.predict_proba(X_test)
    print(predicted_y_val)
      #Value Assignement 
    pred_0 = predicted_y_val[:, 0]
    pred_1 = predicted_y_val[:, 1]
    pred_2 = predicted_y_val[:, 2]
    pred_3 = predicted_y_val[:, 3]
    pred_4 = predicted_y_val[:, 4]
    pred_5 = predicted_y_val[:, 5]
    #Storing them in the Submission file
    submission["High Cost"] = pred_0 
    submission["Higher Cost"] = pred_1
    submission["Highest Cost"] = pred_2
    submission["Low Cost"] = pred_3
    submission["Lower Cost"] = pred_4
    submission["Normal Cost"]= pred_5
    #Turning the submission data into a csv
    submission.to_csv("./Input/sub_LGB.csv", index = False)
    return df
if __name__ == "__main__":
    main()