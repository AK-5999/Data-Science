```python
# notebooks/model_rating_example.py (snippet)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data (exported from SQL or read CSV)
df = pd.read_csv('data/zomato_mumbai.csv')

# 2. Basic cleaning
df = df.dropna(subset=['aggregate_rating', 'average_cost_for_two'])
df['aggregate_rating'] = df['aggregate_rating'].astype(float)
df['average_cost_for_two'] = df['average_cost_for_two'].astype(float)

# 3. Feature engineering (simple example)
df['is_expensive'] = (df['average_cost_for_two'] > df['average_cost_for_two'].median()).astype(int)
df['num_cuisines'] = df['cuisines'].fillna('').apply(lambda x: len(x.split(',')))

# 4. Choose features & target
features = ['average_cost_for_two', 'votes', 'is_expensive', 'num_cuisines', 'establishment']
target = 'aggregate_rating'

X = df[features]
y = df[target]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Preprocessing
numeric_features = ['average_cost_for_two', 'votes', 'num_cuisines']
categorical_features = ['establishment']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 7. Pipeline
model = Pipeline(steps=[
    ('preproc', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# 8. Train
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

```
### Modeling Variants & Ideas
- Classification: Bucket aggregate_rating into labels (e.g., Good/Avg/Bad) and use RandomForestClassifier or XGBoost.
- Clustering: Use KMeans on features like cost, rating, votes to find restaurant segments.
- Time Series / Trend: If you have timestamped reviews, model seasonality on ratings or sales.
### 📈 Outcomes & Business Impact
- Actionable locality insights: Identified top localities and cuisines driving traffic and high ratings.
- Pricing intelligence: Median cost-for-two ~₹1300 (example); helped identify value-for-money vs premium pockets.
- Top performers: Found chains/outlets with best ratings and repeat presence — useful for expansion/partnership decisions.
- SQL-first efficiency: Reduced EDA turnaround by running aggregations directly in DB rather than extracting full datasets.
