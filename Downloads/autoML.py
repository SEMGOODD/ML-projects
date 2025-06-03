import autosklearn.classification
import autosklearn.regression
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    df = pd.read_excel("onesignal.xlsx")
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
        'completed_date', 'completed_time', 'is_weekend',
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
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype('int64')
            # Supprime les colonnes totalement vides ou indétectables
        elif pd.api.types.is_numeric_dtype(X[col]):
            continue  # déjà bon
        else:
            # Convertit les objets non numériques en entiers via factorisation
            X[col], _ = pd.factorize(X[col])
            X = X.loc[:, X.notnull().any()]

    # Remplace les valeurs manquantes
    X = X.fillna(-1)

    # Vérification finale
    assert all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]), "Non-numeric dtypes remain in X"


    # Affichage des colonnes encore non-numériques
    non_numeric_cols = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
    if non_numeric_cols:
        print("Colonnes non numériques restantes :", non_numeric_cols)
        print(X[non_numeric_cols].dtypes)

    # Détecte les colonnes non numériques
    non_numeric_cols = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]

    if non_numeric_cols:
        print("Colonnes non numériques restantes :", non_numeric_cols)
        print(X[non_numeric_cols].dtypes)
        print(X[non_numeric_cols].head())
        raise ValueError("Certaines colonnes sont encore non numériques. Corrige-les.")


    """
    print("Colonnes avec type object avant factorize :")
    print(X.select_dtypes(include=['object']).columns.tolist())
    """

    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.regression.AutoSklearnRegressor()
    #automl = autosklearn.classification.AutoSklearnClassifier()

    automl.fit(X_train, y_train)
    
    y_hat = automl.predict(X_test)
    #accuracy = sklearn.metrics.accuracy_score(y_test, y_hat)


    #print(f"MSE: {mse:.4f}")
    #print(f"R²: {r2:.4f}")
    models_with_weights = automl.get_models_with_weights()

    results = []
    for weight, model in models_with_weights:
        y_hat_model = model.predict(X_test)  # Predict using the individual model
        mse_model = mean_squared_error(y_test, y_hat_model)
        r2_model = r2_score(y_test, y_hat_model)
        results.append((r2_model, mse_model, model))

    # Sort results by r2 descending
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)

    print("\nMeilleurs modèles (triés par r² décroissant) :")
    for r2, mse, model in results_sorted:
        print(f"r2: {r2:.4f} mse: {mse:.4f} - Modèle: {model}")


#docker run -it --rm -m 4g -v C:/Users/HP/Downloads:/app -w /app mfeurer/auto-sklearn:master /bin/bash
"""
    if accuracy > 0.9:
        print("Accuracy score:", accuracy)
        print("Models found:")
        print(automl.show_models())
    else:
        print(f"Accuracy too low: {accuracy:.4f}")
"""