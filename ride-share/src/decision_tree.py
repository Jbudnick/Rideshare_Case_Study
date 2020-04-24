from main import clean_data, plot_feat_importances
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


if __name__ == '__main__':
    data = clean_data(pd.read_csv("data/churn_train.csv"), ['last_trip_date', 'signup_date'], thresh_dict={"driver": 5, "passenger": 5})
    y = data.pop('churn')
    X = data
    X = X.drop('last_trip_date', axis=1)
    X = X.drop('signup_date', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 8, min_samples_leaf = 2)
    clf = clf.fit(X_train, y_train)
    feature_names = X_train.columns
    plot_feat_importances(clf, feature_names, 'images/decisiontree.png')

    #print(cross_val_score(clf, X_train, y_train, cv=10))

    precision =  precision_score(y_test, clf.predict(X_test))
    recall = recall_score(y_test, clf.predict(X_test))
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f'precision : {precision}')
    print(f'recall : {recall}')
    print(f'accuracy : {accuracy}')