import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('/Users/apple/Desktop/msc_project/unsw_nb15_training.csv')

# Display basic information about the dataset
print("Dataset Loaded. Shape:", data.shape)

# Encoding target labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['attack_cat'])

# Separating features (X) and target (y)
X = data.drop(columns=['label', 'attack_cat'])  # Drop label and attack category
Y = data['label']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Initialize classifiers
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Bagging': BaggingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'MLP Neural Network': MLPClassifier(max_iter=300),
    'Balanced Bagging': BalancedBaggingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Easy Ensemble': EasyEnsembleClassifier()
}

# Evaluate classifiers
results = []

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        'Classifier': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")

# Summarize results
results_df = pd.DataFrame(results)
print("\nPerformance Summary:")
print(results_df)

# Selecting top-performing classifier
best_classifier = results_df.loc[results_df['F1-Score'].idxmax()]
print("\nTop-performing Classifier:")
print(best_classifier)
