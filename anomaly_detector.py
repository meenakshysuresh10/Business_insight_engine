from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df[['value']])
    df['anomaly'] = model.predict(df[['value']])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 = normal, -1 = anomaly
    return df, model
