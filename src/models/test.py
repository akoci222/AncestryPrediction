from joblib import load
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

test_features = load('./model/test_ft.pkl')
test_label = load('./model/test_label.pkl')

model = [
    ('Random Forest', './model/randomforest_model.pkl'),
    ('SVMLinear', './model/svmlinear_model.pkl'),
    ('SVMPoly', './model/svmpoly_model.pkl'),
    ('SVMRBF', './model/svmrbf_model.pkl'),
    ('SVMSigmoid', './model/svmsigmoid_model.pkl'),
    ('XGBoost', './model/xgboost_model.pkl')
]

result = []
for mdl_name, mdl_path in model:
    model = load(mdl_path)
    prediction = model.predict(test_features)
    
    balanced_accuracy = balanced_accuracy_score(test_label, prediction)
    report = classification_report(test_label, prediction)
    cm = confusion_matrix(test_label, prediction)
    
    result.append({
        'Model': mdl_name,
        'Accuracy': balanced_accuracy,
        'Classification Report': report,
        'Confusion Matrix': cm
    })

for res in result:
    print(f"Model: {res['Model']}")
    print(f"Accuracy: {res['Accuracy']:.4f}")
    print(f"Classification Report:\n\n{res['Classification Report']}")
    print(f"Confusion matrix:\n\n", res['Confusion Matrix'])



