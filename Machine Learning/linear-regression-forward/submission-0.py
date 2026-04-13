import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # Compute Y_hat = X @ weights
        result = X @ weights
        return np.round(result, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute MSE
        diff = model_prediction - ground_truth
        squared_diff = np.square(diff)
        mse = np.mean(squared_diff)
        return round(mse, 5)