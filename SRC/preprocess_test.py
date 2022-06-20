import preprocess_train
import config
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv(config.TEST)
    df = df.drop(["Tour_ID"], axis=1)
    df["cost_category"] = " "
    df = pd.get_dummies(df, columns=["travel_with", "main_activity", "info_source", "purpose"])
    preprocess_train.remove_nulls(df)
    print(df.isna().sum())
    preprocess_train.encoding(df)
    
    #preprocess_train.scaling(df)
    preprocess_train.sqrt(df)
    #preprocess_train.log_transforms(df)
  
    
    #df = df.drop(["night_zanzibar", "night_mainland", "total_female","total_male"], axis=1)
    print(df.head())
    print(df.describe)
    
    df.to_csv("./Input/preprocessed_test_data.csv", index = False)
    return df

if __name__ == "__main__":
    main()