import pandas as pd
df = pd.read_csv(r"D:\housing.csv")
# print(df.head())
# print(df.shape)
print(df.isnull().sum())
df["total_bedrooms"].fillna(df["total_bedrooms"].mean(),inplace=True)
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# print(df.describe())
# print(df.columns)

X=df.drop("median_house_value",axis=1)
y=df["median_house_value"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test)
print("Predicted Output:",y_pred)
print("Actual Output:",y_test.values)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

sample = [[-120, 35, 20, 3000, 500, 800, 400, 4.5, 1, 0, 0, 0]]
sample = scaler.transform(sample)
print("Pridicted OutPut for sample:",model.predict(sample))

import joblib
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(X.columns)

