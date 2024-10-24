
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.sklearn_wrap import Lasso


REGRESSION_MODELS = [
    "Multiple Linear Regression",
    "Polynomial Regression",
    "Lasso Regression"
] # add your models as str here

CLASSIFICATION_MODELS = [
    "K Nearest Neighbors",
    "Linear SVC",
    "Multinomial Logistic Regression"
] # add your models as str here

def get_model(model_name: str) -> Model:
    if model_name not in REGRESSION_MODELS or CLASSIFICATION_MODELS:
        print(f"No such model `{model_name}` found.")

    # Determining the type of the model
    if model_name in REGRESSION_MODELS:
        type_ = "regression"
    else:
        type_ = "classification"

    # Instantiating the model
    if model_name == "Multiple Linear Regression":
        model = MultipleLinearRegression(type=type_)

    if model_name == "Lasso":
        model = Lasso(type=type_)

    return model
