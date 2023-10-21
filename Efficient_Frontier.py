# loading libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import yfinance as yfin
from dateutil.relativedelta import relativedelta
from datetime import date
today = date.today()
yfin.pdr_override()
# portfolio of stocks parsing
tickers = ['MSFT', 'PEP', 'ACN', 'C', 'META','OMV.VI', 'UBSG.SW', 'ZURN.SW', 'BMW.DE', '6758.T', '^GSPC']
#loading into a new DF
portfolio = pd.DataFrame()
for ticker in tickers:
    portfolio[ticker] = web.get_data_yahoo(ticker,start = '2022-10-20', end = '2023-10-20')['Adj Close']
# changing column names
column_names = ['Microsoft', 'Pepsico', 'Accenture', 'Citigroup', 'Meta', 'OMV', 'UBS', 'Zurich_Insurance_Group', 'BMW', 'Sony', 'S&P500']
portfolio.columns = column_names
print(portfolio.head(5))
# checking the length of a DF
print(len(portfolio))
# checking null values
print(portfolio.isna().sum())
# finding companies with missing data
missing_values = portfolio.isna().sum() > 0
comp_miss_val = portfolio.columns[missing_values]
for i in comp_miss_val:
    portfolio[i] = portfolio[i].fillna(method='ffill')
# no missing data anymore
print(portfolio.isna().sum())
for i in comp_miss_val:
    portfolio[i] = portfolio[i].fillna(method='bfill')
print(portfolio.isna().sum())
# we don't need S&P 500 as of now
portfolio = portfolio.iloc[:,:-1]
print(portfolio.head(2))
# finding returns of stocks
returns = portfolio/portfolio.shift(1) - 1
print(returns)
# finding expected returns 1y interval
expected_returns = returns.mean() * 252
print(expected_returns)
# building a covariance matrix
covariance_matrix = returns.cov() * 252
print(covariance_matrix)
# building a correlation matrix
correlation_matrix = returns.corr()
print(correlation_matrix)
# obtaining number of assets in portfolio (without S&P 500
num_of_assets = len(tickers) - 1
print(num_of_assets)
# creating two random weights
weights = np.random.random(num_of_assets)
weights /= np.sum(weights)
print(weights)
# calculating expected portfolio return
exp_port_return = np.sum(weights * expected_returns)
print(exp_port_return)
# calculating expected portfolio variance
exp_port_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
print(exp_port_variance)
# calculating portfolio SD
port_risk = np.sqrt(exp_port_variance)
print(port_risk)
# simulating 10000 different portfolio construction options
portfolio_returns = []
portfolio_risks = []
for i in range(10000):
    weights = np.random.random(num_of_assets)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(weights * expected_returns))
    portfolio_risks.append(np.sqrt(np.dot(weights.T,np.dot(covariance_matrix, weights))))
# converting to numpy arrays
portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
# checking the number of simulations
print(portfolio_returns[:10])
print(len(portfolio_returns))
print(portfolio_risks[:10])
print(len(portfolio_risks))
# creating a dataframe to combine returns and risks of simulations together
portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_risks})
print(portfolios.head(5))
# building an Effecient Frontier
# loading a matplotlib library
import matplotlib.pyplot as plt
portfolios.plot(x = 'Volatility', y = 'Return', kind = 'scatter', color = 'lightblue')
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
# showing the graph
plt.show()