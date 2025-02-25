import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import traceback
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from joblib import dump
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

model_type = "xgboost"
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
y_ori = df['signal']
le = LabelEncoder()
y_np = le.fit_transform(y_ori)
y = pd.Series(y_np)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

X_train_scaled = X_train.values
X_test_scaled = X_test.values

start_time = time.time()
print(f"Hyperparameter tuning...")

param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6],
    'n_estimators': [50, 100]
}

tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X_train_scaled))
best_score = -np.inf
best_params = None

total_iterations = len(ParameterGrid(param_grid)) * tscv.n_splits
with tqdm(total=total_iterations, desc="Tuning hyperparameters") as pbar:
    for params in ParameterGrid(param_grid):
        scores = []
        for fold_idx, (train_index, test_index) in enumerate(splits, 1):
            X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            classes = np.unique(y_train_fold)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_fold)
            weights_dict = dict(zip(classes, weights))
            sample_weights = [weights_dict[label] for label in y_train_fold]

            dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold, weight=sample_weights)
            dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)

            n_estimators = params['n_estimators']
            params_xgb = params.copy()
            params_xgb.pop('n_estimators')

            model = xgb.train(params_xgb, dtrain_fold, num_boost_round=n_estimators)

            y_pred_fold = model.predict(dval_fold)
            y_pred_fold = np.round(y_pred_fold).astype(int)
            score = f1_score(y_val_fold, y_pred_fold, average='weighted')
            scores.append(score)
            pbar.update(1)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        pbar.set_postfix(best_score=best_score, best_params=best_params)

print(f"\nHyperparameter tuning completed in {time.time() - start_time:.2f} seconds")
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weights_dict = dict(zip(classes, weights))
sample_weights = [weights_dict[label] for label in y_train]

dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weights)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

final_model = xgb.train(best_params, dtrain, num_boost_round=best_params['n_estimators'])

dump(final_model, model_filename)
print(f"Model saved to: {model_filename}")

y_pred_test = final_model.predict(dtest)
y_pred_test = np.round(y_pred_test).astype(int)
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