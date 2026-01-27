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

# Fill missing values if any
df.fillna(0, inplace=True)

# ===== DATA ANALYSIS & VISUALIZATION =====

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

# 4️⃣ Bike rentals by Holiday
if 'holiday' in df.columns:
    plt.figure(figsize=(6,4))
    df.groupby('holiday')['count'].mean().plot.bar(color='orange')
    plt.title("Average Bike Rentals: Holiday vs Non-Holiday")
    plt.xlabel("Holiday")
    plt.ylabel("Average Count")
    plt.show()

# 5️⃣ Scatter plots for key features
for col, color in zip(['temp', 'humidity', 'windspeed'], ['green', 'blue', 'purple']):
    if col in df.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(df[col], df['count'], color=color, alpha=0.5)
        plt.title(f"{col.capitalize()} vs Bike Rentals")
        plt.xlabel(col.capitalize())
        plt.ylabel("Count")
        plt.show()

# 6️⃣ Correlation heatmap (numeric only)
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

# ===== DATA PREPARATION FOR MACHINE LEARNING =====
y = df['count']

# Convert categorical columns to numeric
X = pd.get_dummies(df.drop('count', axis=1), drop_first=True)
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)

print("\nSample of feature data (X):")
print(X.head())

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# ===== MODEL TRAINING =====
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed!")

# ===== PREDICTION =====
y_pred = model.predict(X_test)
print("\nFirst 10 predictions vs actual values:")
print(pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_pred[:10]}))

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

# ===== CUSTOM INPUT PREDICTION =====
print("\n=== Predict Bike Rentals for Custom Input ===")
try:
    temp = float(input("Enter temperature (0-1 normalized): "))
    humidity = float(input("Enter humidity (0-1 normalized): "))
    windspeed = float(input("Enter windspeed (0-1 normalized): "))
    holiday = int(input("Enter holiday (0=No, 1=Yes): "))
    season_spring = int(input("Enter season_spring (1 if spring else 0): "))
    season_summer = int(input("Enter season_summer (1 if summer else 0): "))
    season_winter = int(input("Enter season_winter (1 if winter else 0): "))

    # Create input DataFrame
    input_df = pd.DataFrame([{
        'temp': temp,
        'humidity': humidity,
        'windspeed': windspeed,
        'holiday': holiday,
        'season_spring': season_spring,
        'season_summer': season_summer,
        'season_winter': season_winter
    }])

    # Add missing columns if any
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[X.columns]

    pred_count = model.predict(input_df)[0]
    print(f"\nPredicted bike rentals: {int(pred_count)}")

except Exception as e:
    print("Error in custom input prediction:", e)
