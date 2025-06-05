import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def load_config():
    current_year = datetime.now().year
    holidays = pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31", freq='W-SUN').tolist()
    return {
        'holidays': holidays,
        'prediction_window_days': 7
    }

def load_all_data(data_dir):
    all_dfs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.xlsx') and fname[:4].isdigit():
            df = pd.read_excel(os.path.join(data_dir, fname))
            df['Year'] = int(fname[:4])
            all_dfs.append(df)
    if not all_dfs:
        raise ValueError("No data files found.")
    return pd.concat(all_dfs, ignore_index=True)

def create_features(df, holidays):
    for col in ['A Mech b/d', 'B  Elec b/d', 'N  Planned S/Down']:
        if col not in df.columns:
            df[col] = 0

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df.sort_values(['Machine', 'Date'], inplace=True)

    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['IsHoliday'] = df['Date'].isin(holidays).astype(int)

    #df['Breakdown'] = ((df['A Mech b/d'] + df['B  Elec b/d']) >= 2).astype(int)
    # Ensure breakdown columns are numeric
    df['A Mech b/d'] = pd.to_numeric(df['A Mech b/d'], errors='coerce').fillna(0)
    df['B  Elec b/d'] = pd.to_numeric(df['B  Elec b/d'], errors='coerce').fillna(0)

    # Create binary breakdown target
    df['Breakdown'] = ((df['A Mech b/d'] + df['B  Elec b/d']) >= 2).astype(int)


    for window, smooth in zip([7, 14, 30], [3, 5, 7]):
        df[f'Rolling_{window}'] = df.groupby('Machine')['Breakdown'].transform(lambda x: x.rolling(window, min_periods=1).sum())
        df[f'Rolling_{window}_SMA_{smooth}'] = df.groupby('Machine')[f'Rolling_{window}'].transform(lambda x: x.rolling(smooth, min_periods=1).mean())

    df['LastPM'] = df.groupby('Machine')['N  Planned S/Down'].transform(
        lambda x: x.replace(0, np.nan).fillna(method='ffill')
    )
    df['DaysSincePM'] = df.groupby('Machine')['Date'].transform(
        lambda x: (x - x.mask(df['N  Planned S/Down'] > 0).ffill()).dt.days
    ).fillna(999)

    df['DaysSinceBreakdown'] = df.groupby('Machine')['Date'].transform(
        lambda x: (x - x.mask(df['Breakdown'] > 0).ffill()).dt.days
    ).fillna(999)

    df['BreakdownAfterPM_7d'] = ((df['DaysSincePM'] <= 7) & (df['Breakdown'] == 1)).astype(int)

    df['CumulativeBreakdownHours'] = df[['A Mech b/d', 'B  Elec b/d']].sum(axis=1)
    df['CumulativeBreakdownHours'] = df.groupby('Machine')['CumulativeBreakdownHours'].cumsum()

    return df.dropna(subset=['Date'])

def train_models(X, y):
    if y.nunique() < 2:
        raise ValueError("Insufficient class variety in target variable. Need both 0 and 1 for classification.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    models = {
        'LogisticRegression': Pipeline([
            ('imputer', imputer),
            ('model', LogisticRegression(max_iter=1000))
        ]),
        'RandomForest': Pipeline([
            ('imputer', imputer),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('imputer', imputer),
            ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('imputer', imputer),
            ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
    }

    print("\n📊 Model Accuracy Summary:\n")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = (preds == y_test).mean()
        print(f"{name} Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds))

    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LogisticRegression(max_iter=1000))
    ])
    stack = StackingClassifier(estimators=list(models.items()), final_estimator=final_pipeline, passthrough=True)
    stack.fit(X_train, y_train)

    print("\n✅ Stacked Model Performance:")
    preds = stack.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Stacked Model Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    return stack, models

def predict_next_week(df, model, features, config, selected_machine):
    last_date = datetime.now().date()
    future_dates = [last_date + timedelta(days=i) for i in range(1, config['prediction_window_days'] + 1)]

    last_row = df[df['Machine'] == selected_machine].iloc[-1:]

    if last_row.empty or any(pd.isna(last_row[features].values[0])):
        print(f"⚠️ No valid data available for Machine {selected_machine}.")
        return pd.DataFrame()

    all_predictions = []
    for date in future_dates:
        row = last_row.copy()
        row['Date'] = pd.to_datetime(date)
        row['DayOfWeek'] = date.weekday()
        row['Month'] = date.month
        row['IsHoliday'] = int(date in config['holidays'])
        for w in [7, 14, 30]:
            row[f'Rolling_{w}'] = row[f'Rolling_{w}'].values[0]
            row[f'Rolling_{w}_SMA_{[3,5,7][[7,14,30].index(w)]}'] = row[f'Rolling_{w}_SMA_{[3,5,7][[7,14,30].index(w)]}'].values[0]

        last_actual_date = pd.to_datetime(last_row['Date'].values[0]).date()
        days_ahead = (date - last_actual_date).days

        row['DaysSincePM'] = row['DaysSincePM'].values[0] + days_ahead
        row['DaysSinceBreakdown'] = row['DaysSinceBreakdown'].values[0] + days_ahead
        row['CumulativeBreakdownHours'] = row['CumulativeBreakdownHours'].values[0]
        all_predictions.append(row)

    pred_df = pd.concat(all_predictions)
    pred_df['PredictedBreakdown'] = model.predict(pred_df[features])

    return pred_df[['Date', 'Machine', 'PredictedBreakdown']]

def run_forecast():
    config = load_config()
    data = load_all_data(data_dir='.')
    df = create_features(data, holidays=config['holidays'])

    features = [
        'DayOfWeek', 'Month', 'IsHoliday',
        'Rolling_7', 'Rolling_7_SMA_3',
        'Rolling_14', 'Rolling_14_SMA_5',
        'Rolling_30', 'Rolling_30_SMA_7',
        'DaysSincePM'
    ]
    X = df[features]
    y = df['Breakdown']

    print("Breakdown Class Distribution:")
    print(y.value_counts())

    model, all_models = train_models(X, y)

    print("\n🗓️  Weekly Breakdown Forecast Summary For All Machines:\n")
    all_machines = df['Machine'].unique()
    all_forecasts = []

    for machine in all_machines:
        future_df = predict_next_week(df, model, features, config, machine)
        if not future_df.empty:
            all_forecasts.append(future_df)

    final_df = pd.concat(all_forecasts)
    print(final_df.sort_values(['Date', 'Machine']))

    output_filename = "BreakdownForecast_AllMachines.xlsx"
    final_df.to_excel(output_filename, index=False)
    print(f"\n📁 Forecast saved to: {output_filename}")

if __name__ == '__main__':
    run_forecast()
