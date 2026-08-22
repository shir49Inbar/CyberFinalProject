"""
This file is responsible for the actual traffic classification over encrypted messaging apps.
this module implements a MLP using PyTorch to classify user actions (text, photo, idle) based on network metadata
extracted from datasets exisiting online and actual traffic we created manually.

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from data_handling import FEATURE_COLUMNS


class TrafficDataset(Dataset):
    """
    A custom PyTorch Dataset for loading network traffic features.
    Arguments:
    - features- the extracted metadata features.
    - labels- the numerical labels corresponding to the traffic classes.
    """

    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a single sample and its corresponding label by index.
        """
        return self.X[idx], self.y[idx]


class TrafficMLP(nn.Module):
    """
    MLP architechture for enctypted traffic classification.

    The network consists of 3 fully connected layers with ReLU activations and 
    Dropout for regularization to prevent Overfitting on session-specific network artifacts.
    Arguments:
    - input_size- the number of features in the input data.
    - num_classes- the number of output classes to predict.
    """

    def __init__(self, input_size, num_classes):
        super(TrafficMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        """
        return self.network(x)


def split_by_session(df, test_session=None):
    """
    Splits the dataset into training and testing sets based in session IDs.
    Arguments:
    - df- the complete dataset containing all sessions.
    - test_session: the specific session_id to hold out for testing. 
                    If None, the chronological last session is used.
    Return:
    - tuple: (train_df, test_df, held_out_session_id)
    """
    sessions = sorted(df["session_id"].astype(str).unique())
    if len(sessions) < 2:
        raise ValueError(
            "At least two independent sessions are required to avoid data leakage.")

    test_session = test_session or sessions[-1]
    train = df[df["session_id"].astype(str) != test_session].copy()
    test = df[df["session_id"].astype(str) == test_session].copy()

    return train, test, test_session


def train_model(features_path, test_session=None):
    """
    training and evaluating the MLP model.

    This function handles data loading, preprocessing, target encoding, dataloader initialization, the training loop 
    and the final evaluation using sklearn's classification report.
    Arguments:
    - features_path: path to the aggregated features CSV file.
    - test_session: the session ID to reserve for testing.
    """

    # Load and split data
    df = pd.read_csv(features_path)
    train_df, test_df, held_out_session = split_by_session(df, test_session)

    # Encode the labels into integers
    labels_unique = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(labels_unique)}

    train_labels = train_df["label"].map(label_to_idx).values
    test_labels = test_df["label"].map(label_to_idx).values

    # Preprocessing- to handle missing values and scale features
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_features = scaler.fit_transform(
        imputer.fit_transform(train_df[FEATURE_COLUMNS]))
    test_features = scaler.transform(
        imputer.transform(test_df[FEATURE_COLUMNS]))

    # Initialize Dataloaders
    train_dataset = TrafficDataset(train_features, train_labels)
    test_dataset = TrafficDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the neural network, loss function and optimizer
    input_size = len(FEATURE_COLUMNS)
    num_classes = len(labels_unique)
    model = TrafficMLP(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 50
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log the training progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(y_batch.numpy())

    print(f"\nHeld-out test sessions: {held_out_session}")
    unique_labels_present = np.unique(np.concatenate([all_targets, all_preds]))
    present_names = [labels_unique[i]
                     for i in unique_labels_present if i < len(labels_unique)]
    print(classification_report(all_targets, all_preds,
          target_names=present_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds, labels=unique_labels_present)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{name}" for name in present_names], columns=[
                         f"Predicted_{name}" for name in present_names])
    print(cm_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a MLP classifier on encrypted messaging metadata."
    )
    parser.add_argument(
        "features", help="Path to the features CSV generated by data_handling.py")
    parser.add_argument(
        "--test-session", help="Entire session ID reserved for testing (e.g., S002)")
    args = parser.parse_args()

    train_model(args.features, args.test_session)
