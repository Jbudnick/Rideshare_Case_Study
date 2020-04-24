from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from main import clean_data, plot_feat_importances
import pandas as pd

if __name__ == '__main__':
    
    data = pd.read_csv("data/churn_train.csv")
    data = clean_data(data, ['last_trip_date', 'signup_date'], thresh_dict={"driver": 5, "passenger": 5})
    data.drop(columns=['last_trip_date', 'signup_date'], inplace=True)
    headers = data.drop(columns=['churn']).columns
    X = np.asarray(data.drop(columns=['churn']))
    y = np.asarray(data['churn'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    model = GradientBoostingClassifier(
                                        learning_rate=0.2,
                                        random_state=2,
                                        n_estimators=300,
                                        max_leaf_nodes=20
                                    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:,1]
    thresh=0.51
    y_preds = (y_pred_proba >= thresh).astype(int)
    print("\nCV")
    print("Accuracy:\t", accuracy_score(y_test, y_preds))
    print("Precision:\t", precision_score(y_test, y_preds))
    print("Recall:\t\t", recall_score(y_test, y_preds))

    plot_feat_importances(model, headers, "images/feature_importances_gbc.png")

    print("\nHoldout")

    model = GradientBoostingClassifier(
                                        learning_rate=0.2,
                                        random_state=2,
                                        n_estimators=300,
                                        max_leaf_nodes=20
                                    )
    
    model.fit(X, y)
    holdout = clean_data(pd.read_csv("data/churn_test.csv"), ['last_trip_date', 'signup_date'], thresh_dict={"driver": 5, "passenger": 5})
    holdout.drop(columns=['last_trip_date', 'signup_date'], inplace=True)
    X_holdout = np.asarray(holdout.drop(columns=['churn']))
    y_holdout= np.asarray(holdout['churn'])

    y_pred_proba = model.predict_proba(X_holdout)[:,1]
    thresh=0.51
    y_preds = (y_pred_proba >= thresh).astype(int)

    print("Accuracy:\t", accuracy_score(y_holdout, y_preds))
    print("Precision:\t", precision_score(y_holdout, y_preds))
    print("Recall:\t\t", recall_score(y_holdout, y_preds))