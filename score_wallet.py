import json
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(filepath):
    with open(filepath) as f:
        raw = json.load(f)
    records = []
    for entry in raw:
        flat = entry.copy()
        flat.update(entry.get('actionData', {}))
        records.append(flat)
    return pd.DataFrame(records)

def preprocess(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['action'] = df['action'].str.lower()
    return df

def feature_engineering(df):
    features = []
    for addr, group in df.groupby('userWallet'):
        deposits = group[group.action == 'deposit']
        borrows = group[group.action == 'borrow']
        repays = group[group.action == 'repay']
        redeems = group[group.action == 'redeemunderlying']
        liquidations = group[group.action == 'liquidationcall']

        total_deposit = deposits.amount.sum()
        total_borrow = borrows.amount.sum()
        total_repay = repays.amount.sum()
        total_redeem = redeems.amount.sum()
        n_borrows = len(borrows)
        n_liquidations = len(liquidations)
        n_txns = len(group)

        repayment_ratio = total_repay / total_borrow if total_borrow > 0 else 1.0
        liquidation_ratio = n_liquidations / n_borrows if n_borrows > 0 else 0.0
        deposit_borrow_ratio = total_deposit / total_borrow if total_borrow > 0 else 1.0
        activity_level = np.log1p(n_txns)

        features.append({
            'userWallet': addr,
            'total_deposit': total_deposit,
            'total_borrow': total_borrow,
            'total_repay': total_repay,
            'total_redeem': total_redeem,
            'n_borrows': n_borrows,
            'n_liquidations': n_liquidations,
            'n_txns': n_txns,
            'repayment_ratio': repayment_ratio,
            'liquidation_ratio': liquidation_ratio,
            'deposit_borrow_ratio': deposit_borrow_ratio,
            'activity_level': activity_level,
        })
    return pd.DataFrame(features)

def train_model(features_df):
    np.random.seed(42)
    features_df['true_score'] = (
        0.3 * features_df['repayment_ratio'] +
        0.2 * (1 - features_df['liquidation_ratio']) +
        0.3 * features_df['deposit_borrow_ratio'] +
        0.2 * (features_df['activity_level'] / features_df['activity_level'].max())
    ) + np.random.normal(0, 0.05, len(features_df))

    features_df['true_score'] = MinMaxScaler((300, 1000)).fit_transform(features_df[['true_score']])
    X = features_df.drop(columns=['userWallet', 'true_score'])
    y = features_df['true_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    features_df['predicted_score'] = model.predict(X).clip(0, 1000).round(2)
    return features_df[['userWallet', 'predicted_score']]

def analyze_scores(score_df, plot_path='score_distribution.png'):
    bins = list(range(0, 1100, 100))
    labels = [f"{b}-{b+100}" for b in bins[:-1]]
    score_df = score_df[['userWallet', 'predicted_score']].copy()
    score_df.loc[:, 'score_bin'] = pd.cut(score_df['predicted_score'], bins=bins, labels=labels, right=False)

    bin_counts = score_df['score_bin'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    bin_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Wallet Credit Score Distribution')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def main(json_path):
    print("Loading data...")
    df = load_data(json_path)
    df = preprocess(df)
    print("Extracting features...")
    features = feature_engineering(df)
    print("Training machine learning model...")
    scores = train_model(features)
    scores.to_csv("wallet_scores.csv", index=False)
    print("Generating score distribution plot...")
    analyze_scores(scores)
    print("Score distribution saved to score_distribution.png")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python score_wallet.py <json_file_path>")
    else:
        main(sys.argv[1])
