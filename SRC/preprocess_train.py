from numpy import NaN
import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler



def main():
    df = pd.read_csv(config.TRAIN)
    df =  df.drop(["Tour_ID"], axis=1)
    encoding(df)
    print(df.describe())
    remove_nulls(df)
    sqrt(df)
    #scaling(df)
    #log_transforms(df)
    df = pd.get_dummies(df, columns=["travel_with","main_activity", "info_source","purpose"])#attempting for country name
    #df = df.drop(["night_zanzibar", "night_mainland", "total_female","total_male"], axis=1)
    print(df.head())
    print(df.dtypes)
    print(df.isna().sum())
    print(df.describe())
    df.to_csv("./Input/preprocessed_train_data.csv", index=False)
    return df

#Data cleaning
#Encoding packages
def remove_nulls(df):
    #travel_with, total_female and total_male have nulls
    #Imputing total_female and total_male with means
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp = imp.fit(df[["total_female","total_male"]])
    df[["total_female", "total_male"]] = imp.transform(df[["total_female","total_male"]])
    #Dealing with nulls in [travel_with]
    #df["cost_category"].hist()
    #plt.show
    return df

def log_transforms(df):
    #total_male log
    df["total_male"] = df["total_male"] + 1
    df["total_male"] = np.log(df["total_male"])
    #total female log
    #cost category log
    df["cost_category"] = df["cost_category"]+1
    df["cost_category"] = np.log(df["cost_category"]) 
    return df

def scaling(df):
    scaler = MinMaxScaler()
    scaling = df[["night_zanzibar", "night_mainland", "total_male", "total_female"]]
    scaling_scaled = scaler.fit_transform(scaling)
    scaling_scaled =  pd.DataFrame(scaling_scaled)
    df["night_zanzibar"] = scaling_scaled[0]
    df["night_mainland"] = scaling_scaled[1]
    df["total_male"] = scaling_scaled[2]
    df["total_female"] = scaling_scaled[3]
    print(scaling_scaled)
    return df
def sqrt(df):
    df["total_male"] = np.sqrt(df["total_male"])
    df["total_female"] = np.sqrt(df["total_female"])
    df["night_zanzibar"] = np.sqrt(df["night_zanzibar"])
    df["night_mainland"] = np.sqrt(df["night_mainland"])
    df["cost_category"] = np.sqrt(df["cost_category"])
    return df
def encoding(df):
    le =  LabelEncoder()
    categ = ["package_transport_int","package_accomodation","package_food","package_transport_tz","package_sightseeing","tour_arrangement", "package_guided_tour", "package_insurance", "first_trip_tz"]
    df[categ] = df[categ].apply(le.fit_transform)
    df[categ] = df[categ].astype(float)
    rem = ["night_mainland", "night_zanzibar"]
    df[rem] = df[rem].astype(float)
    #Manually encoding labels to improve performance
    #Encoding "Travel_With"
    #enc_travel = {'With Children':0, 'With Spouse':1, 'With Spouse and Children':2, 'Alone':3, 'With Other Friends/Relatives':4}
    #df["travel_with"] =  df.travel_with.map(enc_travel)
    #df["travel_with"] = df["travel_with"].astype(float)
    #Manually encoding "Cost"
    cost_dict =  {'High Cost':0, 'Higher Cost':1,'Highest Cost':2,'Low Cost':3,'Lower Cost':4,'Normal Cost':5}
    df["cost_category"] =  df.cost_category.map(cost_dict)
    df["cost_category"] =  df["cost_category"].astype(float) #int type to allow xgboost to work simultaneously
    #Manually encoding age
    age_dict =  {'<18':0,'18-24':1, '25-44':2, '45-64':3, '65+':4}
    df["age_group"] = df.age_group.map(age_dict)
    df["age_group"] = df["age_group"].astype(float)
    #Frequency Encoding
    #fe_co = df.groupby("country").size()/len(df)
    #df.loc[:, "country_freq_encoded"] = df["country"].map(fe_co)
    #Probability  Ratio Encoding 
    return df

if __name__ == '__main__':
    main()      