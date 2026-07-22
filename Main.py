import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = yf.download("TSLA", period="6mo", interval="1d")
data.columns = data.columns.get_level_values(0)

data['lag_1'] = data['Close'].shift(1)
data['lag_2'] = data['Close'].shift(2)
data['lag_3'] = data['Close'].shift(3)

data = data.dropna()
X = data[['lag_1', 'lag_2', 'lag_3']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Average prediction error: ${error:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label="Actual Dollar Volume")
plt.plot(y_test.index, predictions, label="Predicted Dollar Volume")
plt.title("TSLA: Actual vs Predicted Daily Dollar Volume")
plt.xlabel("Date")
plt.ylabel("Dollar Volume (USD)")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()