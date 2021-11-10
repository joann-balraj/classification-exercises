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


def prep_telco(df):
    df = df.drop_duplicates()
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    df = df.drop(columns =cols_to_drop)
    dummy_df = pd.get_dummies(df[['gender', 'contract_type', 'internet_service_type', 'payment_type']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df