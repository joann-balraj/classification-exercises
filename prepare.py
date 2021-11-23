import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire

def prep_iris(df):
    cols_to_drop = ['species_id']
    df = df.drop(columns = cols_to_drop)
    df.rename(columns={"species_name": "species"}, inplace = True)
    dummy_df = pd.get_dummies(df[['species']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df


def prep_titanic(df):
    '''
    This function will clean the data...
    '''
    df = df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    df = df.drop(columns =cols_to_drop)
    df['embark_town'] = df.embark_town.fillna(value = 'Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test

def split_titanic_data(df):
    
    #Combines the clean_titanic_data, split_titanic_data, and impute_mean_age functions.
    
    df = prep_titanic(df)

    train, validate, test = split_titanic_data(df)
    
    train, validate, test = impute_mean_age(train, validate, test)

    return train, validate, test


def prep_telco(df):
    df = df.drop_duplicates()
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    df = df.drop(columns =cols_to_drop)
    dummy_df = pd.get_dummies(df[['gender', 'contract_type', 'internet_service_type', 'payment_type']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df