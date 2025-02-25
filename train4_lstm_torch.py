import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import time
import traceback
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

model_type = "lstm"
with open("train.yaml", "r") as f:
    config = yaml.safe_load(f)

filepath = config[model_type]['filepath']
n_features = config[model_type]['n_features']
feature_columns = config[model_type]['feature_columns']
best_model_path = config[model_type]['best_model_path']
start_date = config[model_type]['start_date']
end_date = config[model_type]['end_date']
epochs = config[model_type]['epochs']
batch_size = config[model_type]['batch_size']
hidden_dim = config[model_type]['hidden_dim']

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
    df = df.loc[start_date:end_date]
    data = df[feature_columns].values

    print(f"Creating sliding window")
    strides = data.strides[0]
    X = as_strided(data, shape=(len(data) - n_features + 1, n_features, len(feature_columns)), strides=(strides, strides, data.strides[1]))

    print(f"Preparing target variable")
    y = df['signal'][n_features - 1:]
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = torch.tensor(y, dtype=torch.long)
    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")

except Exception as e:
    print(f"Data preprocessing error: {e}")
    traceback.print_exc()
    exit()

train_size = 0.7
val_size = 0.2

n_samples = len(X)
n_train = int(n_samples * train_size)
n_val = int(n_samples * val_size)

X_train = X[:n_train]
y_train = y[:n_train]

X_val = X[n_train:n_train + n_val]
y_val = y[n_train:n_train + n_val]

X_test = X[n_train + n_val:]
y_test = y[n_train + n_val:]

print(f"Dataset split: Train 7, Validation 2, Test 1, in {time.time() - start_time:.2f} seconds")

scaler = StandardScaler()

X_train = torch.tensor(scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape), dtype=torch.float32)
X_val = torch.tensor(scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape), dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_dim = X_train.shape[2]
output_dim = len(le.classes_)
model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)

y_train = np.asarray(y_train)
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters())

scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3)
best_val_f1 = 0.0

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/lstm_experiment_{current_time}'
writer = SummaryWriter(log_dir=log_dir)

for epoch in range(epochs):
    model.train()
    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{epochs}") as pbar:
        train_loss = 0.0
        train_true = []
        train_preds = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())

            train_loss += loss.item()
            pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        train_report = classification_report(le.inverse_transform(train_true), le.inverse_transform(train_preds), output_dict=True, zero_division=1)
        current_train_f1 = train_report['weighted avg']['f1-score']

        print(f"\nEpoch {epoch + 1} Training Results:")
        print(f" Weighted Avg F1: {current_train_f1:.4f}, Accuracy: {train_report['accuracy']:.4f}, Avg Loss: {avg_train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_preds = []
        val_true = []
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_outputs_batch = model(X_val_batch)
            _, predicted_batch = torch.max(val_outputs_batch, 1)

            val_preds.extend(predicted_batch.cpu().numpy())
            val_true.extend(y_val_batch.cpu().numpy())

            loss = criterion(val_outputs_batch, y_val_batch)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_report = classification_report(le.inverse_transform(val_true), le.inverse_transform(val_preds), output_dict=True, zero_division=1)
        current_val_f1 = val_report['weighted avg']['f1-score']

        print(f"\nEpoch {epoch + 1} Validation Results:")
        print(f" Weighted Avg F1: {current_val_f1:.4f}, Accuracy: {val_report['accuracy']:.4f}, Avg Loss: {avg_val_loss:.4f}")

        scheduler.step(current_val_f1)
        current_lr = scheduler.get_last_lr()
        print(f"Current learning rate: {current_lr}")

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}, F1: {best_val_f1}")

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('F1/Train', current_train_f1, epoch)
    writer.add_scalar('F1/Validation', current_val_f1, epoch)
    writer.add_scalar('Learning Rate', current_lr[0], epoch)

writer.close()

best_model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
best_model.load_state_dict(state_dict)

best_model.eval()
with torch.no_grad():
    test_preds = []
    test_true = []
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
        test_outputs_batch = best_model(X_test_batch)
        _, predicted_batch = torch.max(test_outputs_batch, 1)
        test_preds.extend(predicted_batch.cpu().numpy())
        test_true.extend(y_test_batch.cpu().numpy())

    print("\nTest Results (using best model):")
    print(classification_report(le.inverse_transform(test_true), le.inverse_transform(test_preds), zero_division=1))

    cm = confusion_matrix(test_true, test_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    labels = sorted(list(set(test_true)))
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

print("Training and evaluation finished")
input("View Tensorboard: ")