import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def train_baseline_model(df):

    def extract_label(filename):
        if 'image' in filename.lower():
            return 'Image'
        if 'audio' in filename.lower() or 'voice' in filename.lower():
            return 'Voice'
        if 'text' in filename.lower():
            return 'Text'
        return 'Unknown'

    df['label'] = df['source_file'].apply(extract_label)

    df_clean = df[df['label'] != 'Unknown']

    features = ['packet_count', 'total_bytes', 'mean_packet_size',
                'std_packet_size', 'mean_iat', 'uplink_ration']
    X = df_clean[features]
    y = df_clean['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluating model
    y_pred = rf_model.predict(X_test_scaled)

    print("== Classification Report ==")
    print(classification_report(y_test, y_pred))

    importances = pd.Series(rf_model.feature_importances_,
                            index=features).sort_values(ascending=False)
    print("\n===Feature Importance ===")
    print(importances)

    return rf_model, y_test, y_pred
