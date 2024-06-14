import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # Load the dataset
    try:
        data = pd.read_csv("kc_house_data.csv")
    except FileNotFoundError:
        print("The file 'kc_house_data.csv' was not found. Please ensure the file is in the correct directory.")
        return

    # Display the first few rows and describe the dataset
    print(data.head())
    print(data.describe())

    # Exploratory Data Analysis (EDA)
    plt.figure(figsize=(10, 6))
    data['bedrooms'].value_counts().plot(kind='bar')
    plt.title('Number of Bedrooms')
    plt.xlabel('Bedrooms')
    plt.ylabel('Count')
    sns.despine()
    plt.show()

    plt.figure(figsize=(10, 10))
    sns.jointplot(x=data.lat.values, y=data.long.values, height=10)
    plt.ylabel('Longitude', fontsize=12)
    plt.xlabel('Latitude', fontsize=12)
    plt.show()
    sns.despine()

    plt.scatter(data.price, data.sqft_living)
    plt.title("Price vs Square Feet")
    plt.xlabel("Price")
    plt.ylabel("Square Feet")
    plt.show()

    plt.scatter(data.price, data.long)
    plt.title("Price vs Location of the Area")
    plt.xlabel("Price")
    plt.ylabel("Longitude")
    plt.show()

    plt.scatter(data.price, data.lat)
    plt.xlabel("Price")
    plt.ylabel('Latitude')
    plt.title("Latitude vs Price")
    plt.show()

    plt.scatter(data.bedrooms, data.price)
    plt.title("Bedrooms and Price")
    plt.xlabel("Bedrooms")
    plt.ylabel("Price")
    plt.show()
    sns.despine()

    plt.scatter((data['sqft_living'] + data['sqft_basement']), data['price'])
    plt.title("Total Square Feet (Living + Basement) vs Price")
    plt.xlabel("Total Square Feet")
    plt.ylabel("Price")
    plt.show()

    plt.scatter(data.waterfront, data.price)
    plt.title("Waterfront vs Price (0 = no waterfront)")
    plt.xlabel("Waterfront")
    plt.ylabel("Price")
    plt.show()

    # Prepare the data for modeling
    labels = data['price']
    features = data.drop(['id', 'price', 'date'], axis=1)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, random_state=2)

    # Train a Linear Regression model
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred_linear = reg.predict(x_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    print(f'Linear Regression Mean Squared Error: {mse_linear}')
    print(f'Linear Regression R^2 Score: {reg.score(x_test, y_test)}')

    # Train a Gradient Boosting Regressor model
    clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='squared_error')
    clf.fit(x_train, y_train)
    y_pred_gbr = clf.predict(x_test)
    mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    print(f'Gradient Boosting Regressor Mean Squared Error: {mse_gbr}')
    print(f'Gradient Boosting Regressor R^2 Score: {clf.score(x_test, y_test)}')

    # Visualize the Gradient Boosting Regressor performance
    params = {'n_estimators': 400}
    t_sc = np.zeros((params['n_estimators']), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(x_test)):
        t_sc[i] = mean_squared_error(y_test, y_pred)

    testsc = np.arange((params['n_estimators'])) + 1

    plt.figure(figsize=(12, 6))
    plt.plot(testsc, clf.train_score_, 'b-', label='Training Set Deviance')
    plt.plot(testsc, t_sc, 'r-', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()

    # Perform PCA (optional)
    scaler = StandardScaler()
    train1_scaled = scaler.fit_transform(features)
    pca = PCA()
    train1_pca = pca.fit_transform(train1_scaled)
    print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')

    plt.figure(figsize=(10, 8))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

if __name__ == "__main__":
    main()
