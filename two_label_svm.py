import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from preprocess import df_combined

class CustomSVMMixed:
    def __init__(self, svc_params=None, svr_params=None, class_weights=None):
        self.svc_params = svc_params 
        self.svr_params = svr_params
        self.svc = SVC(**self.svc_params, class_weight=class_weights)  # For categorical variable
        self.svr = SVR(**self.svr_params)  # For continuous variable
        self.label_encoder = LabelEncoder()

    def fit(self, X, y_cat, y_cont):
        self.svc.fit(X, y_cat)

        # Encode addiction labels numerically
        y_cat_encoded = self.label_encoder.fit_transform(y_cat)

        X_copy = X.copy()
        # Append encoded addiction level to features for SVR
        X_copy['Addiction_Level_Encoded'] = y_cat_encoded

        self.svr.fit(X_copy, y_cont)
        return self

    def predict(self, X):
        y_cat_pred = self.svc.predict(X)

        # Encode predicted labels for use in SVR
        y_cat_pred_encoded = self.label_encoder.transform(y_cat_pred)

        X_copy= X.copy()
        # Append predicted addiction level to features for SVR prediction
        X_copy["Addiction_Level_Encoded"] = y_cat_pred_encoded

        # Predict productivity score
        y_cont_pred = self.svr.predict(X_copy)
        return y_cat_pred, y_cont_pred


X = df_combined.iloc[:, :-3]  
y_cat = df_combined.iloc[:, -2]  # Addiction Class (categorical)
y_cont = df_combined.iloc[:, -1]  # Productivity score (continuous)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_cat_train, y_cat_test, y_cont_train, y_cont_test = train_test_split(
    X_scaled_df, y_cat, y_cont, test_size=0.25, random_state=42
)

class_weights = {
    'Low': 1,
    'Medium': 3,
    'High': 3
}

# Initialize and train the custom SVM model
model = CustomSVMMixed(
    svc_params={'kernel': 'rbf', 'C': 1.0, 'probability': True},  # For categorical
    svr_params={'kernel': 'rbf', 'C': 1.0, 'gamma' : 0.01},  # For continuous
    class_weights=class_weights
)
model.fit(X_train, y_cat_train, y_cont_train)

# Predict on test set
y_cat_pred, y_cont_pred = model.predict(X_test)


unique_labels = np.unique(y_cat)
# Evaluate
print("\nClassification Report for Addiction Level:")
print(classification_report(y_cat_test, y_cat_pred, labels=unique_labels))
print("Addiction Level Accuracy:", accuracy_score(y_cat_test, y_cat_pred))
print("Productivity Score MSE:", mean_squared_error(y_cont_test, y_cont_pred))



# Confusion matrix
cm = confusion_matrix(y_cat_test, y_cat_pred, labels=unique_labels)
cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
print("\nConfusion Matrix for Addiction Level:")
print(cm_df)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Addiction Level")
plt.show()

# Evaluate continuous target: MSE and Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_cont_test, y_cont_pred, alpha=0.5)
plt.plot([y_cont_test.min(), y_cont_test.max()], [y_cont_test.min(), y_cont_test.max()], 'r--', lw=2)
plt.xlabel("Actual Productivity Score")
plt.ylabel("Predicted Productivity Score")
plt.title("Actual vs Predicted for productivity Score")
plt.tight_layout()
plt.show()


#XAI


# For classifier
result_svc = permutation_importance(model.svc, X_test, y_cat_test, n_repeats=10, random_state=42)
svc_importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': result_svc.importances_mean})
print("\nSVC Feature Importance:")
print(svc_importance_df.sort_values(by='Importance', ascending=False))

# For regressor
# Prepare input with predicted addiction class
y_cat_pred_encoded = model.label_encoder.transform(y_cat_pred)
X_test['Addiction_Level_Encoded'] = y_cat_pred_encoded

result_svr = permutation_importance(model.svr, X_test, y_cont_test, n_repeats=10, random_state=42)

# Include 'Addiction_Level_Encoded' in feature names
svr_feature_names = list(X_test.columns)
svr_importance_df = pd.DataFrame({'Feature': svr_feature_names, 'Importance': result_svr.importances_mean})
print("\nSVR Feature Importance:")
print(svr_importance_df.sort_values(by='Importance', ascending=False))


fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
axes[0].barh(svc_importance_df['Feature'], svc_importance_df['Importance'], color='skyblue')
axes[0].set_title('SVC Feature Importance')
axes[0].set_xlabel('Importance Score')
axes[0].invert_yaxis()

# SVR Plot
axes[1].barh(svr_importance_df['Feature'], svr_importance_df['Importance'], color='lightgreen')
axes[1].set_title('SVR Feature Importance')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()