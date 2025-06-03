
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    df = pd.read_excel("C://Users/HP/Downloads/onesignal.xlsx")
    threshold = len(df) * 0.5
    cols_to_drop = [col for col in df.columns if df[col].count() < threshold and col != "email_reply_to_address"]
    df = df.drop(columns=cols_to_drop)
    df = df.drop(columns=["email_click_tracking_disabled"])

    df['include_unsubscribed'] = df['include_unsubscribed'].fillna(0)
    df = df.drop(columns=["include_unsubscribed"])

    df['completed_date'] = df['completed_date'].ffill()
    df['completed_time'] = df['completed_time'].ffill()
    df['completed_date_id'] = df['completed_date_id'].ffill()
    df['included_segments'] = df['included_segments'].ffill()

    df = df.drop(columns=["app_id", "isEmail", "frequency_capped", "email_subject"])

    df['queued_datetime'] = pd.to_datetime(df['queued_date'].astype(str) + ' ' + df['queued_time'].astype(str))
    df['send_after_datetime'] = pd.to_datetime(df['send_after_date'].astype(str)+ ' ' + df['send_after_time'].astype(str), errors='coerce')
    df['completed_datetime'] = pd.to_datetime(df['completed_date'].astype(str) + ' ' + df['completed_time'].astype(str), errors='coerce')

    df['delay_minutes'] = (df['send_after_datetime'] - df['queued_datetime']).dt.total_seconds() / 60
    df['send_duration_minutes'] = (df['completed_datetime'] - df['send_after_datetime']).dt.total_seconds() / 60
    df['total_pipeline_minutes'] = (df['completed_datetime'] - df['queued_datetime']).dt.total_seconds() / 60

    df['send_hour'] = df['send_after_datetime'].dt.hour
    df['send_dayofweek'] = df['send_after_datetime'].dt.dayofweek
    df['is_weekend'] = df['send_dayofweek'] >= 5

    df.drop(columns=[
        'queued_date', 'queued_time', 'send_after_date', 'send_after_time',
        'completed_date', 'completed_time',
        'queued_date_id', 'send_after_date_id', 'completed_date_id'
    ], inplace=True)

    cols_to_encode = [
        'notification_id', 'email_from_name', 'email_from_address', 'email_preheader',
        'email_reply_to_address', 'included_segments', 'excluded_segments',
        'delayed_option', 'delivery_time_of_day'
    ]

    for col in cols_to_encode:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col], _ = pd.factorize(df[col])

    X = df.drop(columns=["converted"])
    y = df["converted"]

    # Convertir tout X en valeurs numériques
    for col in X.columns:
        print(col, X[col].dtype)
        if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col], _ = pd.factorize(X[col])    