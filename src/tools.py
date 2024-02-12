from sklearn import linear_model
from sklearn import model_selection
import pandas

def elastic_net_regression(predictor, target, test_size = 0.2, random_state = None):
    
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(predictor, target, test_size = test_size, random_state = random_state)
    
    elastic_net = linear_model.LinearRegression()
    elastic_net.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = elastic_net.predict(X_train)
    
    y_train = 10**y_train.squeeze()
    y_pred = 10**y_pred.squeeze()

    # Calculate the errors
    errors = y_train - y_pred

    # Create a DataFrame with y_test, y_pred, and errors
    df = pandas.DataFrame({'Actual': y_train, 'Predicted': y_pred, 'Error': errors})
    
    print("Elastic Net Regression score: ", elastic_net.score(X_test, y_test))
    
    print("Elastic Net Regression coefficients: \n", elastic_net.coef_)
    
    print("Elastic Net Regression intercept: ", elastic_net.intercept_)
    
    print("Elastic Net residuals info: \n", df.describe())
    
    