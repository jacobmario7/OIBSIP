import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sp= pd.read_csv("C:\\Users\\Jacob Mario Leonard\\Downloads\\advertising.csv")

X = sp[['TV', 'Radio', 'Newspaper']]
y = sp['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_


y_pred = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Best Model Parameters:", best_params)
print("Mean Squared Error:", mse)

