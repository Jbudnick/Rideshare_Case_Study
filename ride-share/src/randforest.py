from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy import interp
from sklearn.metrics import roc_curve, auc
from main import clean_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
plt.style.use('ggplot')


def plot_roc(X, y, clf_class, plot_name, **kwargs):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' %
                 (i, roc_auc))
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} Receiver operating characteristic'.format(plot_name))
    plt.legend(loc="lower right")
    plt.savefig('ROC{}'.format(plot_name))
    plt.show()

def plot_feat_importances(model, feature_names, out_filepath):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(feature_names[indices][:5], importances[indices][:5], color="blue")
    ax.set_title(f"Feature importances - {type(model).__name__}")
    ax.set_xlabel("Feature", fontsize=16)
    ax.set_ylabel("Feature importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_filepath)
    plt.show()
    return

if __name__ =='__main__':

    churn_df = pd.read_csv('data/churn.csv')
    churn_test_df = pd.read_csv('data/churn_test.csv')
    churn_train_df = pd.read_csv('data/churn_train.csv')

    churn_df = clean_data(churn_df, ['last_trip_date', 'signup_date'])
    churn_test_df = clean_data(churn_test_df, ['last_trip_date', 'signup_date'])
    churn_train_df = clean_data(churn_train_df, ['last_trip_date', 'signup_date'])

    churn_train_df.drop(['last_trip_date', 'signup_date'],
                        axis=1, inplace=True)

    churn_test_df.drop(['last_trip_date', 'signup_date'],
                        axis=1, inplace=True)

    y = churn_train_df.pop('churn').values
    X = churn_train_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_hat = rf.predict(X_test)

    print("Accuracy store of RF is", rf.score(X_test, y_test))
    print(confusion_matrix(y_test, y_hat))
    print('Precision: {}, Recall: {}'.format(
    precision_score(y_test, y_hat), recall_score(y_test, y_hat)))

    plot_roc(X, y, RandomForestClassifier, plot_name='Random_Forest')

    plot_feat_importances(rf, churn_train_df.columns,
                          'RandomForestImportances')

    #Test with holdout data
    y_true = churn_test_df.pop('churn').values
    X_f_test = churn_test_df.values

    y_predict = rf.predict(X_f_test)
    
    print("Holdout Data: Accuracy store of RF is", rf.score(X_f_test, y_true))
    print(confusion_matrix(y_true, y_predict))
    print('Holdout Data: Precision: {}, Recall: {}'.format(
        precision_score(y_true, y_predict), recall_score(y_true, y_predict)))
