import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



def main():
    avocado_file_path = 'data/avocado.csv'
    data = pd.read_csv(avocado_file_path)
    data = data.drop('Unnamed: 0',axis = 1)

    print("\nData head:")
    print(data.head())

    # Data information
    print("\nDataset Information:")
    print(data.info())

    print("\nDescriptive Statistics:")
    print(data.describe())

    print("\nColumns in the Dataset:")
    print(data.columns)

    # * Data Preprocessing
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is in datetime format
    data = data.dropna()  # Drop missing values
    data = data.sort_values('Date')  # Ensure data is sorted by date

    data = pd.get_dummies(data, columns=['region', 'type'], drop_first=True)

    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    X = data.drop(columns=['AveragePrice', 'Date'])
    y = data['AveragePrice']

    # * Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # * Model Training 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # * Model Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # * Visualize Predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs. Predicted Avocado Prices')
    plt.grid(True)
    plt.show()

    # * Feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    feature_importances.head(min(10, len(feature_importances))).plot(kind='bar', color='skyblue')
    plt.title('Top 10 Feature Importances')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.show()

    # * Forecasting
    from forecasting import avocado_forecasting
    print("\nPerforming Time Series Forecasting...")
    forecast_data = data[['Date', 'AveragePrice']].copy()  
    forecast_data.set_index('Date', inplace=True)  
    forecast_data = forecast_data.resample('M').mean().fillna(method='ffill')  

    forecast = avocado_forecasting(forecast_data)
    print(f"Forecast for next 12 months:\n{forecast}")


if __name__ == "__main__":
    main()
