import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt

#Assuming Churn is the positive (Churn = 1)

# Surge percent and avg surge may be correlated; may need to drop one or the other

# NaN values:
#   Convert phones to dummy values - only 3 values Baseline: Unknown

#   Missing ratings:
#       Or classify ratings as high, medium, low, unknown instead of numerical floats low = below 4, medium between 4 and 5, high = 5, unknown = NaN
#           Set up in a way that can be easily changed
#       Take average of values to fill NaNs for - dummy variable for Nan or not?
#       Trying to predict churn; how to handle missing ratings

#   Clean_data function - Sam 

# Taylor and Allison - EDA


#churn_df['avg_rating_by_driver'].hist()

# Random Forest
# Gradient Boosted Trees
# Logistic Regression - Need standardization
# MLP 


def clean_data(data, col_list):
    cleaned_df = data.copy()
    cleaned_df = pd.get_dummies(data, columns=['city'], drop_first = True)
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
