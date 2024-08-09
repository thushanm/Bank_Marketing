import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_models(X_train, y_train, X_test, y_test):
    # SVM Model
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    # Logistic Regression Model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Evaluation
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
    print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

if __name__ == "__main__":
    df = pd.read_csv('../data/bank_marketing_pca.csv')
    y = pd.read_csv('../data/bank_marketing_preprocessed.csv')['y']
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    train_models(X_train, y_train, X_test, y_test)
