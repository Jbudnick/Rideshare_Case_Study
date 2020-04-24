import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt

#Assuming Churn is the positive (Churn = 1)


def clean_data(data, col_list):
    cleaned_df = data.copy()
    cleaned_df = pd.get_dummies(data, columns=['city'])
    cleaned_df = pd.get_dummies(data, columns=['city'])
    for col in col_list:
        cleaned_df[col] = pd.to_datetime(churn_df[col])
    cleaned_df['churn'] = cleaned_df['last_trip_date'] < date_30_days_ago()
    cleaned_df['churn'].replace({True: 1, False: 0}, inplace=True)
    cleaned_df['luxury_car_user'].replace({True: 1, False: 0}, inplace=True)
    return cleaned_df

def date_30_days_ago():
    current_date = datetime.strptime('2014-07-01', '%Y-%m-%d')
    earliest_date = current_date - dt.timedelta(days=30)
    return earliest_date


if __name__ == '__main__':
    churn_df = pd.read_csv('data/churn.csv')
    churn_test_df = pd.read_csv('data/churn_test.csv')
    churn_train_df = pd.read_csv('data/churn_train.csv')

    churn_df = clean_data(churn_df, ['last_trip_date', 'signup_date'])
    churn_test_df = clean_data(churn_test_df, ['last_trip_date', 'signup_date'])
    churn_train_df = clean_data(churn_train_df, ['last_trip_date', 'signup_date'])
    print(churn_df.head())
