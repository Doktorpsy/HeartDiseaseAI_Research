
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

df = pd.read_csv('data/heart_statlog_cleveland_hungary_final.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "LogisticRegression": Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=5000))]),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": Pipeline([('scaler',StandardScaler()),('model',SVC(probability=True))])
}

results=[]
best_model=None
best_auc=0

for name, model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,1]

    auc=roc_auc_score(y_test,prob)

    results.append({
        "Model":name,
        "Accuracy":accuracy_score(y_test,pred),
        "Precision":precision_score(y_test,pred),
        "Recall":recall_score(y_test,pred),
        "F1":f1_score(y_test,pred),
        "ROC_AUC":auc
    })

    if auc>best_auc:
        best_auc=auc
        best_model=model

pd.DataFrame(results).sort_values("ROC_AUC",ascending=False).to_csv(
    "results/model_comparison.csv",index=False
)

joblib.dump(best_model,"results/best_model.pkl")
print("Training complete.")
