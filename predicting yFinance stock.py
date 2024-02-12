from lib2to3.pgen2.pgen import DFAState
from random import betavariate
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Define a list of stock symbols
stock = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Fetch historical data
data = yf.download(stock, start='2022-01-01', end='2022-12-31')

# Data will be loaded into a DataFrame 'data'
print(data.head(5))
print(data.describe)
print(data.columns)
print(data.size)

# Calculate daily returns for each stock
daily_returns = data['Adj Close'].pct_change()


# Calculate the average daily return for each stock
average_daily_return = daily_returns.mean()


# Calculate the volatility for each stock
volatility = daily_returns.std()

# calculate the correlation matrix
correlation_matrix = daily_returns.corr()


# Display the results
print("Average Daily Returns:")
print(average_daily_return)
print("\nVolatility:")
print(volatility)
print("\nCorrelation Matrix:")
print(correlation_matrix)

sma_window = 50
sma = data['Adj Close'].rolling(sma_window).mean()
print(sma)

# Calculate the exponential moving average (EMA) for each stock with a 50-day window
ema = data['Adj Close'].ewm(span=sma_window, adjust=False).mean()
print(ema)
# Generate buy/sell stock based on SMA and EMA crossovers
buy_stock = data['Adj Close'] > sma
sell_stock = data['Adj Close'] < sma

# Display the results
print("Simple Moving Average (SMA):")
print(sma.tail())

print("\nExponential Moving Average (EMA):")
print(ema.tail())

"""
print(buy_stock)
print(sell_stock)
print(data['Adj Close'] > ema)
print(data['Adj Close'] < ema)

"""
print("\nBuy/Sell Signals:")

signals = pd.DataFrame({
    'Buy Stock (SMA)': [buy_stock],
    'Sell Stock (SMA)': [sell_stock],
    'Buy Stock (EMA)': [data['Adj Close'] > ema],
    'Sell Stock (EMA)': [data['Adj Close'] < ema],
})
print(signals.tail(5))

# Define the weights of each stock in the portfolio
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# Calculate the covariance matrix of daily returns
cov_matrix = daily_returns.cov()

# Calculate the expected return of the portfolio
portfolio_return = np.dot(weights, average_daily_return)

# Calculate the portfolio's risk (volatility)
portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

# Display the results
print("Expected Return of the Portfolio:", portfolio_return)
print("Portfolio Risk (Volatility):", portfolio_risk)

#Creating the visualization

plt.figure(figsize=(10, 6))
for i, stock in enumerate(stock):
    stock_return = average_daily_return[stock]
    stock_risk = daily_returns[stock].std()
    plt.scatter(stock_risk, stock_return, label=stock)

# Plot the risk-return profile of the portfolio
plt.scatter(portfolio_risk, portfolio_return, color='red', marker='o', label='Portfolio')

plt.title('Risk-Return Profile')
plt.xlabel('Risk (Volatility)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()


for stock in stock:
    plt.plot(data.index, data['Adj Close'][stock], label=stock)

plt.title('Historical Stock Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()



# Define the risk-free rate (you can replace this with the actual risk-free rate)
risk_free_rate = 0.02  # Example risk-free rate of 2%

# Calculate the portfolio's daily excess returns (returns above the risk-free rate)
portfolio_excess_returns = (daily_returns.dot(weights) - risk_free_rate).dropna()
print(portfolio_excess_returns)

# Calculate the Sharpe ratio
sharpe_ratio = (portfolio_excess_returns.mean() / portfolio_excess_returns.std()) * np.sqrt(252)  # Assuming 252 trading days in a year
print(sharpe_ratio)

# Calculate the Sortino ratio
downside_returns = portfolio_excess_returns[portfolio_excess_returns < 0]
sortino_ratio = (portfolio_excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

# Calculate Jensen's alpha (also known as the Jensen's measure)
#benchmark_return = data['^GSPC'].pct_change().mean()  # Assuming S&P 500 as the benchmark
#alpha = portfolio_excess_returns.mean() - (risk_free_rate + betavariate * (benchmark_return - risk_free_rate))

# Print the results
print("Sharpe Ratio:", sharpe_ratio)
print("Sortino Ratio:", sortino_ratio)
#print("Jensen's Alpha:", alpha)

targeted_value = data['Adj Close']
print(targeted_value)


risk_tolerance = 0.15  # Example: 15% annual risk tolerance
investment_horizon = 5  # Example: 5-year investment horizon
"""
if sharpe_ratio > 0.5:
    recommendation = "Consider a moderate allocation to this portfolio."
elif sharpe_ratio > 0:
    recommendation = "Consider a conservative allocation to this portfolio."
else:
    recommendation = "Consider other investment options with better risk-adjusted returns."

print("Expected Return of the Portfolio:", portfolio_return)
print("Portfolio Risk (Volatility):", portfolio_risk)
print("Sharpe Ratio:", sharpe_ratio)
print("Investment Recommendation:", recommendation)
"""
X = data[['SMA_50', 'SMA_200', 'Volume', 'Daily_Return', 'sharpe_ratio', 'sortino_ratio']]

# Define the dependent variable (target)
y = data[targeted_value]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
#print(f"Mean Squared Error: {mse}")

# Provide investment recommendations based on the model's predictions
if y_pred[-1] > y_test[-1]:
    recommendation = "Consider buying this stock."
else:
    recommendation = "Consider selling or holding this stock."

print("Investment Recommendation:", recommendation)