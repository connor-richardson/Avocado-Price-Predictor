import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    #Load data
    avocado_file_path = 'data/avocado.csv'
    data = pd.read_csv(avocado_file_path)

    #Data information
    print("\nDataset Information:")
    print(data.info())

    print("\nFirst Five Rows:")
    print(data.head())

    print("\nDescriptive Statistics:")
    print(data.describe())

    print("\nColumns in the Dataset:")
    print(data.columns)


    # Step 3: Data Preprocessing
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is in datetime format
    data = data.dropna()  # Drop missing values


    # Encode categorical variables (e.g., 'region', 'type')
    data = pd.get_dummies(data, columns=['region', 'type'], drop_first=True)


   # Step 4: Feature Engineering
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    # Define features and target variable
    X = data.drop(columns=['AveragePrice', 'Date'])
    y = data['AveragePrice']

    # Step 5: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Model Training (Random Forest Regressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 7: Model Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Step 8: Visualize Predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs. Predicted Avocado Prices')
    plt.grid(True)
    plt.show()

    # Step 9: Visualize Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    feature_importances.head(10).plot(kind='bar', color='skyblue')
    plt.title('Top 10 Feature Importances')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.show()


if __name__ == "__main__":
    main()
