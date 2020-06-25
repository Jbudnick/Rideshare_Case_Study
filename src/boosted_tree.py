import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import clean_data, plot_feat_importances

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array

        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='neg_mean_squared_error')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best

if __name__ == "__main__":
    churn_df = pd.read_csv('data/churn.csv')
    churn_test_df = pd.read_csv('data/churn_test.csv')
    churn_train_df = pd.read_csv('data/churn_train.csv')

    churn_df = clean_data(churn_df, ['last_trip_date', 'signup_date'], thresh_dict={"driver": 5, "passenger": 5})
    churn_test_df = clean_data(churn_test_df, ['last_trip_date', 'signup_date'], thresh_dict={"driver": 5, "passenger": 5})
    churn_train_df = clean_data(churn_train_df, ['last_trip_date', 'signup_date'], thresh_dict={"driver": 5, "passenger": 5})

    churn_train_df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)
    churn_test_df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)

    y = churn_train_df.pop('churn').values
    X = churn_train_df.values

    y_holdout = churn_test_df.pop('churn').values
    X_holdout = churn_test_df.values

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=2)

    abr = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', n_estimators=300, random_state=2, 
                                    min_samples_leaf=20, max_depth=2, max_features=3)
    abr.fit(X_train, y_train)
    y_pred = abr.predict(X_test)
    print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))
    print("AdaBoost Precision:", precision_score(y_test, y_pred))
    print("AdaBoost Recall Score:", recall_score(y_test, y_pred))

    feature_importances = np.argsort(abr.feature_importances_)
    feature_names = list(churn_train_df.columns[feature_importances[-1:-6:-1]])
    print("\n12: Top Five:", feature_names)

    plot_feat_importances(abr, churn_train_df.columns, 'images/adaboost_feat_importance.png')

    y_pred_holdout = abr.predict(X_holdout)
    print("AdaBoost Accuracy:", accuracy_score(y_holdout, y_pred_holdout))
    print("AdaBoost Precision:", precision_score(y_holdout, y_pred_holdout))
    print("AdaBoost Recall Score:", recall_score(y_holdout, y_pred_holdout))
    
