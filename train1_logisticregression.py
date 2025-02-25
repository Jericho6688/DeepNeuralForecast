import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import traceback
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

model_type = "logisticregression"
with open("train.yaml", "r") as f:
    config = yaml.safe_load(f)

filepath = config[model_type]['filepath']
n_features = config[model_type]['n_features']
sentiment_add = config[model_type]['sentiment_add']
feature_columns = config[model_type]['feature_columns']
model_filename = config[model_type]['model_filename']
start_date = config[model_type]['start_date']
end_date = config[model_type]['end_date']

start_time = time.time()
try:
    df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()
    exit()

start_time = time.time()
try:
    all_features = []
    for col in feature_columns:
        values = df[col].values
        strides = values.strides[0]
        features = as_strided(values, shape=(len(df) - n_features, n_features), strides=(strides, strides))
        all_features.append(features)

    feature_cols = []
    for col in feature_columns:
        feature_cols.extend([f'{col}_t-{i + 1}' for i in range(n_features)])

    features_df = pd.DataFrame(np.concatenate(all_features, axis=1), columns=feature_cols, index=df.index[n_features:])
    df = pd.concat([df, features_df], axis=1).dropna()
    df = df.loc[start_date:end_date]
    print(f"\nUsing data from {start_date} to {end_date}")
    print(f"Features created in {time.time() - start_time:.2f} seconds")
    print("\nFeature columns:")
    print(df.columns.tolist())

except Exception as e:
    print(f"Error creating features: {e}")
    traceback.print_exc()
    exit()

columns_to_drop = ['timestamp', 'gmtoffset', 'signal', 'count', 'normalized']
for column in sentiment_add:
    if column in columns_to_drop:
        columns_to_drop.remove(column)
X = df.drop(columns_to_drop, axis=1)
print("\nTraining feature columns:")
print(X.columns.tolist())
y = df['signal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start_time = time.time()
print(f"Hyperparameter tuning...")

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'loss': ['log_loss'],
    'penalty': ['l1', 'l2']
}

tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X_train_scaled))
best_score = -np.inf
best_params = None

classes = np.unique(y_train)

total_iterations = len(ParameterGrid(param_grid)) * tscv.n_splits
with tqdm(total=total_iterations, desc="Tuning hyperparameters") as pbar:
    for params in ParameterGrid(param_grid):
        scores = []
        for fold_idx, (train_index, test_index) in enumerate(splits, 1):
            X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            sgd_classifier = SGDClassifier(random_state=42, max_iter=1000, **params)

            batch_size = 1000
            n_batches = (len(X_train_fold) + batch_size - 1) // batch_size

            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(X_train_fold))
                X_batch = X_train_fold[start:end]
                y_batch = y_train_fold.iloc[start:end]
                sgd_classifier.partial_fit(X_batch, y_batch, classes=classes)

            y_pred_fold = sgd_classifier.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted'))
            pbar.update(1)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        pbar.set_postfix(best_score=best_score, best_params=best_params)

print(f"\nHyperparameter tuning completed in {time.time() - start_time:.2f} seconds")
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")

best_sgd_classifier = SGDClassifier(random_state=42, max_iter=1000, class_weight='balanced', **best_params)
best_sgd_classifier.fit(X_train_scaled, y_train)

dump(best_sgd_classifier, model_filename)
print(f"Best model saved to: {model_filename}")

y_pred_test = best_sgd_classifier.predict(X_test_scaled)
print("\nTest set performance:")
print(classification_report(y_test, y_pred_test, digits=4, zero_division=1))

cm = confusion_matrix(y_test, y_pred_test)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
labels = sorted(list(set(y_test)))

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="Blues", cbar=True,
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 8})

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j + 0.5, i + 0.7, str(cm[i, j]), ha='center', va='center', color='black', fontsize=10)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Test Set (Values and Percentage)')
plt.show()