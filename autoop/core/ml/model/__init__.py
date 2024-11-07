
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.\
    regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.\
    regression.sklearn_wrap import LassoWrapper
from autoop.core.ml.model.\
    regression.polynomial_regression import PolynomialRegression
from autoop.core.ml.model.\
    classification.k_nearest_neighbors import KNearestNeighbors
from autoop.core.ml.model.\
    classification.linear_svc import Linear_SVC
from autoop.core.ml.model.\
    classification.\
        multinomial_logistic_regression import MultinomialLogisticRegression


REGRESSION_MODELS = [
    "Multiple Linear Regression",
    "Polynomial Regression",
    "Lasso Wrapper"
] # add your models as str here

CLASSIFICATION_MODELS = [
    "K Nearest Neighbors",
    "Linear SVC",
    "Multinomial Logistic Regression"
] # add your models as str here

def get_model(model_name: str) -> Model:
    if model_name not in REGRESSION_MODELS and CLASSIFICATION_MODELS:
        print(f"No such model `{model_name}` found.")

    # Determining the type of the model
    if model_name in REGRESSION_MODELS:
        type_ = "regression"
    else:
        type_ = "classification"

    # Instantiating the model
    if model_name == "Multiple Linear Regression":
        model = MultipleLinearRegression(type=type_)
    if model_name == "Lasso Wrapper":
        model = LassoWrapper(type=type_)
    if model_name == "Polynomial Regression":
        model = PolynomialRegression(type=type_)
    if model_name == "K Nearest Neighbors":
        model = KNearestNeighbors(type=type_)
    if model_name == "Linear SVC":
        model = Linear_SVC(type=type_)
    if model_name == "Multinomial Logistic Regression":
        model = MultinomialLogisticRegression(type=type_)

    return model
