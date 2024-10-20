
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression

REGRESSION_MODELS = [
    "MultipleLinearRegression"
] # add your models as str here

CLASSIFICATION_MODELS = [
    "K_nearest_neighbour"
] # add your models as str here

def get_model(model_name: str) -> Model:
    if model_name not in REGRESSION_MODELS or CLASSIFICATION_MODELS:
        print(f"No such model `{model_name}` found.")

    if model_name == "MultipleLinearRegression":
        model = MultipleLinearRegression()

    return model