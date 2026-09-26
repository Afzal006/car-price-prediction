import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("car data.csv")

print("Dataset loaded successfully!")


# ==========================================
# 2. Remove Car_Name
# ==========================================

df = df.drop("Car_Name", axis=1)


# ==========================================
# 3. Features and Target
# ==========================================

X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]


# ==========================================
# 4. Feature Types
# ==========================================

numerical_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]


# ==========================================
# 5. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 6. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 7. Models
# ==========================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10
    )
}


# ==========================================
# 8. Train Models
# ==========================================

results = {}

trained_models = {}

for name, algorithm in models.items():

    print()
    print("=" * 50)
    print("Training:", name)
    print("=" * 50)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", algorithm)
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = model

    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)


# ==========================================
# 9. Model Comparison
# ==========================================

results_df = pd.DataFrame(results).T

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)


# ==========================================
# 10. Select Model
# ==========================================

selected_model_name = "Random Forest"

selected_model = trained_models[selected_model_name]

print()
print("Selected model:", selected_model_name)


# ==========================================
# 11. Create models Folder
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 12. Save Model
# ==========================================

joblib.dump(
    selected_model,
    "models/car_price_model.pkl"
)

print()
print("Model saved successfully!")
print("Location: models/car_price_model.pkl")