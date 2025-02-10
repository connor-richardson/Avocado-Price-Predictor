import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def avocado_forecasting(data):
    # * Check data
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The input data must have a DateTimeIndex.")

    # * Set frequency
    if data.index.freq is None:
        data = data.asfreq('M')  # Monthly default
        print("Frequency was not set. Defaulted to 'M' (monthly).")

    # * Check stationarity with Augmented Dickey-Fuller
    print("\nChecking stationarity with ADF test:")
    result = adfuller(data['AveragePrice'])
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] > 0.05:
        print("Warning: The data is non-stationary (p-value > 0.05). Differencing may be required.")

    # * Plot ACF and PACF to determine p and q for ARIMA
    print("\nPlotting ACF and PACF:")
    max_lags = min(40, len(data) // 2)  # error prevention if data too short
    
    plot_acf(data['AveragePrice'], lags=max_lags)
    plt.title("Autocorrelation (ACF)")
    plt.show()
    
    plot_pacf(data['AveragePrice'], lags=max_lags)
    plt.title("Partial Autocorrelation (PACF)")
    plt.show()

    # * Fit ARIMA model
    print("\nFitting ARIMA model...")
    model = ARIMA(data['AveragePrice'], order=(5, 1, 0))  
    model_fit = model.fit()
    print(model_fit.summary())

    # * Forecasting for next 12 months
    print("\nForecasting the next 12 months...")
    forecast = model_fit.forecast(steps=12)
    print("Forecasted Prices:")
    print(forecast)

    # * Forecast visuals
    forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthEnd(), periods=12, freq='M')
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['AveragePrice'], label='Historical Prices')
    plt.plot(forecast_index, forecast, label='Forecasted Prices', color='red')
    plt.title('Avocado Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast


# if __name__ == "__main__":
#     # For standalone testing
#     avocado_file_path = 'data/avocado.csv'
#     raw_data = pd.read_csv(avocado_file_path)

#     raw_data['Date'] = pd.to_datetime(raw_data['Date'])
#     raw_data = raw_data.sort_values(by='Date')
#     time_series_data = raw_data[['Date', 'AveragePrice']].copy()
#     time_series_data.set_index('Date', inplace=True)
#     time_series_data = time_series_data.resample('M').mean()

#     forecast = avocado_forecasting(time_series_data)
