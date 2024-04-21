import pandas as pd
import joblib
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('./data/processed/model_data_2_pop.csv')
features = data.iloc[:, 1:-2]  # features
label = data.iloc[:, -1]    # target

label_encoder = LabelEncoder()
label = label_encoder.fit_transform(label)

#init models
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
xgbc = XGBClassifier(booster='gbtree', learning_rate=0.3, max_depth=6, alpha=0, min_child_weight=1)
kernel = ['linear', 'poly', 'rbf', 'sigmoid']


skf = StratifiedKFold(n_splits=8, shuffle=True)

for train_index, test_index in skf.split(features, label):
    train_features, test_features = features.iloc[train_index], features.iloc[test_index]
    train_label, test_label = label[train_index], label[test_index]
    #dump(test_features, './model/test_ft.pkl')
    #dump(test_label, './model/test_label.pkl')

    # random forest
    rfc.fit(train_features, train_label)
    rfc_prediction = rfc.predict(test_features)
    rfc_accuracy = balanced_accuracy_score(test_label, rfc_prediction)
    rfc_report = classification_report(test_label, rfc_prediction)
    rfc_cm = confusion_matrix(test_label, rfc_prediction)
    #joblib.dump(rfc, './model/randomforest_model.pkl')

    # xgboost
    xgbc.fit(train_features, train_label)
    xgbc_prediction = xgbc.predict(test_features)
    xgbc_accuracy = balanced_accuracy_score(test_label, xgbc_prediction)
    xgbc_report = classification_report(test_label, xgbc_prediction)
    xgbc_cm = confusion_matrix(test_label, xgbc_prediction)
    #joblib.dump(xgbc, './model/xgboost_model.pkl')


    # svm
    svmc_accuracies = {}
    svmc_cm = {}
    svmc_report = {}
    for k in kernel:
        svmc = SVC(C=1, kernel=k, gamma='scale')
        svmc.fit(train_features, train_label)
        svmc_prediction = svmc.predict(test_features)
        svmc_accuracies[k] = balanced_accuracy_score(test_label, svmc_prediction)
        svmc_report[k] = classification_report(test_label, svmc_prediction)
        svmc_cm[k] = confusion_matrix(test_label, svmc_prediction)
        #joblib.dump(svmc, f'./model/svm{k}_model.pkl')


# print eval metrics
print(f"Random Forest accuracy: {rfc_accuracy:.4f}")
print("Random Forest classification report:\n\n", rfc_report)
print("Random Forest confusion matrix:\n\n", rfc_cm)

print(f"XGBoost accuracy: {xgbc_accuracy:.4f}")
print("XGBoost classification report:\n", xgbc_report)
print("XGBoost confusion matrix:\n", xgbc_cm)

print("\nSVM accuracies:")
for k, accuracy in svmc_accuracies.items():
    print(f"{k}: {accuracy:.4f}")
    print(f"{k} classification report:\n\n", svmc_report[k])
    print(f"{k} confusion matrix:\n\n", svmc_cm[k])
