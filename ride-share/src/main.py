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


def clean_data(data, date_col_list, thresh_dict={"driver": 4, "passenger": 4}):
    cleaned_df = data.copy()
    cleaned_df = pd.get_dummies(cleaned_df, columns=['city'], drop_first = True)

    # convert date_col_list to datetime
    for col in date_col_list:
        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
    
    # create churn col
    cleaned_df['churn'] = cleaned_df['last_trip_date'] < date_30_days_ago('2014-07-01')

    # convert to binary
    cleaned_df['churn'].replace({True: 1, False: 0}, inplace=True)
    cleaned_df['luxury_car_user'].replace({True: 1, False: 0}, inplace=True)

    # fill na
    cleaned_df = pd.get_dummies(cleaned_df, columns=['phone'])

    # ratings thresholds
    driver_rating_thresh = thresh_dict['driver']
    cleaned_df['high_driver_rating'] = (cleaned_df['avg_rating_of_driver'] >= driver_rating_thresh).astype(int)
    cleaned_df['low_driver_rating'] = (cleaned_df['avg_rating_of_driver'] < driver_rating_thresh).astype(int)

    passenger_rating_thresh = thresh_dict['passenger']
    cleaned_df['high_passenger_rating'] = (cleaned_df['avg_rating_by_driver'] >= passenger_rating_thresh).astype(int)
    cleaned_df['low_passenger_rating'] = (cleaned_df['avg_rating_by_driver'] < passenger_rating_thresh).astype(int)

    cleaned_df.drop(columns=['avg_rating_of_driver', 'avg_rating_by_driver'], inplace=True)
    
    return cleaned_df

def date_30_days_ago(date):
    current_date = datetime.strptime(date, '%Y-%m-%d')
    earliest_date = current_date - dt.timedelta(days=30)
    return earliest_date

def plot_feat_importances(model, feature_names, out_filepath):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.bar(feature_names[indices][:5], importances[indices][:5], color="blue")
    ax.set_title(f"Feature importances - {type(model).__name__}")
    ax.set_xlabel("Feature", fontsize=16)
    ax.set_ylabel("Feature importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_filepath)
    plt.close()
    return
    
if __name__ == '__main__':
    churn_df = pd.read_csv('data/churn.csv')
    churn_test_df = pd.read_csv('data/churn_test.csv')
    churn_train_df = pd.read_csv('data/churn_train.csv')

    churn_df = clean_data(churn_df, ['last_trip_date', 'signup_date'])
    churn_test_df = clean_data(churn_test_df, ['last_trip_date', 'signup_date'])
    churn_train_df = clean_data(churn_train_df, ['last_trip_date', 'signup_date'])

    churn_train_df.info()