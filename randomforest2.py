import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
url = "ElectricityBillDataset.csv"
df = pd.read_csv(url)

# Convert 'Month' to numerical values
month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
              'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
df['Month'] = df['Month'].map(month_dict)

# Split the dataset into features (X) and target variable (y)
X = df[['Month', 'Year', 'units_consumed']]
y = df['Electricity_Bill']

# Create a RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)

# Train the model
rf_model.fit(X, y)

# Predict next year's electricity bill
next_year_data = pd.DataFrame({
    'Month': range(1, 13),
    'Year': 2024,  # Replace with the desired year for prediction
    'units_consumed': 198.654545  # Replace with the units consumed for prediction
})

next_year_data['Predicted'] = rf_model.predict(next_year_data[['Month', 'Year', 'units_consumed']])

# Display the predicted bills for each month
for month, bill in zip(range(1, 13), next_year_data['Predicted']):
    month_name = [k for k, v in month_dict.items() if v == month][0]
    print(f'Predicted Electricity Bill for {month_name} 2024: {bill}')

# Calculate and display the accuracy
y_pred = rf_model.predict(X)
accuracy = r2_score(y, y_pred)
accuracy_percentage = accuracy * 100

print(f'\nModel Accuracy (R-squared): {accuracy_percentage:.2f}%')

# Plot actual and predicted bills
plt.plot(df['Month'], df['Electricity_Bill'], label='Actual Bills', marker='o')
plt.plot(next_year_data['Month'], next_year_data['Predicted'], label='Predicted Bills', marker='o')
plt.xlabel('Month')
plt.ylabel('Electricity Bill (in units)')
plt.legend()
plt.title('Actual vs Predicted Electricity Bills')
plt.show()
