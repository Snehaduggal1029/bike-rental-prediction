# ===== IMPORT LIBRARIES =====
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== LOAD DATASET =====
df = pd.read_csv("kaggle_bike_cleaned.csv")

# ===== BASIC DATA CHECK =====
print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# ===== VISUALIZATIONS =====

# 1️⃣ Histogram of bike rentals
plt.figure(figsize=(6,4))
plt.hist(df['count'], bins=30, color='skyblue')
plt.title("Distribution of Bike Rentals")
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Pie chart for Season
if 'season' in df.columns:
    plt.figure(figsize=(5,5))
    df['season'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=['gold', 'skyblue', 'lightgreen', 'pink']
    )
    plt.title("Bike Rentals by Season")
    plt.ylabel("")
    plt.show()

# 3️⃣ Bar chart for Weather
if 'weather' in df.columns:
    plt.figure(figsize=(6,4))
    df['weather'].value_counts().plot.bar(color='lightcoral')
    plt.title("Bike Rentals by Weather")
    plt.xlabel("Weather")
    plt.ylabel("Number of Records")
    plt.show()

# 4️⃣ Bike rentals by Holiday (0 = No, 1 = Yes)
if 'holiday' in df.columns:
    plt.figure(figsize=(6,4))
    df.groupby('holiday')['count'].mean().plot.bar(color='orange')
    plt.title("Average Bike Rentals: Holiday vs Non-Holiday")
    plt.xlabel("Holiday")
    plt.ylabel("Average Count")
    plt.show()

# 5️⃣ Scatter plot: Temperature vs Count
if 'temp' in df.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(df['temp'], df['count'], color='green', alpha=0.5)
    plt.title("Temperature vs Bike Rentals")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Count")
    plt.show()

# 6️⃣ Scatter plot: Humidity vs Count
if 'humidity' in df.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(df['humidity'], df['count'], color='blue', alpha=0.5)
    plt.title("Humidity vs Bike Rentals")
    plt.xlabel("Humidity (%)")
    plt.ylabel("Count")
    plt.show()

# 7️⃣ Scatter plot: Windspeed vs Count
if 'windspeed' in df.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(df['windspeed'], df['count'], color='purple', alpha=0.5)
    plt.title("Windspeed vs Bike Rentals")
    plt.xlabel("Windspeed")
    plt.ylabel("Count")
    plt.show()

# 8️⃣ Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

# ===== DATA PREPARATION =====
y = df['count']  # target variable
X = pd.get_dummies(df.drop('count', axis=1), drop_first=True)  # features

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MODEL TRAINING =====
model = LinearRegression()
model.fit(X_train, y_train)

# ===== PREDICTION =====
y_pred = model.predict(X_test)

# ===== MODEL EVALUATION =====
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n==== MODEL EVALUATION ====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# ===== VISUALIZE PREDICTIONS =====
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='purple', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title("Actual vs Predicted Bike Rentals")
plt.show()
