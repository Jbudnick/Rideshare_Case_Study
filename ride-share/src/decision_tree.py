from main import clean_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
plt.style.use('ggplot')

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
    data = clean_data(pd.read_csv("../data/churn_train.csv"), ['last_trip_date', 'signup_date'], thresh_dict={"driver": 5, "passenger": 5})
    y = data.pop('churn')
    X = data
    X = X.drop('last_trip_date', axis=1)
    X = X.drop('signup_date', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 8, min_samples_leaf = 2)
    clf = clf.fit(X_train, y_train)
    feature_names = X_train.columns
    plot_feat_importances(clf, feature_names, '../images/randomforestreg5.png')

    #print(cross_val_score(clf, X_train, y_train, cv=10))

    precision =  precision_score(y_test, clf.predict(X_test))
    recall = recall_score(y_test, clf.predict(X_test))
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f'precision : {precision}')
    print(f'recall : {recall}')
    print(f'accuracy : {accuracy}')


