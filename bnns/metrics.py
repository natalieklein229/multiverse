
import torch
from matplotlib import pyplot as plt
from bnns.utils import iqr, cor, univar_poly_fit
from quality_of_life.my_base_utils          import my_warn
from quality_of_life.my_visualization_utils import points_with_curves

#
# ~~~ Measure MSE of the predictive median relative 
def mse_of_median( predictions, y_test ):
    with torch.no_grad():
        y_test = y_test.flatten()
        pred = predictions.median(dim=-1).values
        assert pred.shape == y_test.shape
        return (( pred - y_test )**2).mean()

#
# ~~~ Measure MSE of a model on a pair (X,y) of tensors
def mse_of_mean( predictions, y_test ):
    with torch.no_grad():
        y_test = y_test.flatten()
        pred = predictions.mean(dim=-1)
        assert pred.shape == y_test.shape
        return (( pred - y_test )**2).mean()

# #
# # ~~~ Measure MSE of the predictive median relative 
# def median_of_mse( predictions, y_test ):
#     with torch.no_grad():
#         y_test = y_test.flatten()
#         pred = predictions.T
#         assert pred.shape[1] == y_test.shape[0]
#         return (( pred - y_test )**2).mean(dim=0).median().values

# #
# # ~~~ Measure MSE of the predictive median relative 
# def mean_of_mse( predictions, y_test ):
#     with torch.no_grad():
#         y_test = y_test.flatten()
#         pred = predictions.T
#         assert pred.shape[1] == y_test.shape[0]
#         return (( pred - y_test )**2).mean()

#
# ~~~ Measure strength of the relation "predictive uncertainty (std. dev. / iqr)" ~ "accuracy (MSE of the predictive mean / predictive median)"
def uncertainty_vs_accuracy( predictions, y_test, quantile_uncertainty, quantile_accuracy, show=True ):
    with torch.no_grad():
        y_test = y_test.flatten() 
        assert y_test.shape[0] == predictions.shape[0]
        uncertainty = iqr(predictions,dim=-1) if quantile_uncertainty else predictions.std(dim=-1)
        accuracy = (predictions.median(dim=-1).values-y_test)**2 if quantile_accuracy else (predictions.mean(dim=-1)-y_test)**2
        uncertainty  =  uncertainty.cpu().numpy()
        accuracy     =     accuracy.cpu().numpy()
        #
        # ~~~ Use polynomial regression to measure the strength of the relation uncertainty~accuracy
        fits = [ univar_poly_fit( y=uncertainty, x=accuracy, degree=k ) for k in (1,2,3,4) ]
        polys = [ fit[0] for fit in fits ]
        R_squared_coefficients = [ fit[2] for fit in fits ]
        if show:
            points_with_curves(
                    y = uncertainty,
                    x = accuracy,
                    points_label = "Values on the Val./Test Set",
                    marker_color = "grey",
                    curves = polys,
                    curve_labels = [ f"R^2 {val:.3}" for val in R_squared_coefficients ],
                    title  =  "Uncertainty vs Accuracy of Predictions (with Polynomial Fits)",
                    xlabel = "Accuracy of Predictions (viz. |True Value - Posertior Predictive " + ("median" if quantile_accuracy else "mean") + "|^2)",
                    ylabel = "Uncertainty (viz. Posterior Predictive " + "IQR)" if quantile_uncertainty else "std. dev.)",
                    model_fit = False
                )
        slope_in_OLS = fits[0][1][0]
        R_squared_of_OLS = R_squared_coefficients[0]
        if (max(R_squared_coefficients) > R_squared_of_OLS+0.1):
            my_warn(f"The OLS fit (R^2 {R_squared_of_OLS:.4}) was outperformed by a higher degree polynomial fit (R^2 {max(R_squared_coefficients):.4}). Using either beta_1 from OLS or the correlation may result in an inaccurate quantification of the relation. Please check the plot returned by the `show=True` argument.")
        return slope_in_OLS, cor(uncertainty,accuracy)

#
# ~~~ Measure strength of the relation "predictive uncertainty (std. dev.)" ~ "distance from training points"
def uncertainty_vs_proximity( predictions, y_test, quantile_uncertainty, x_test, x_train, show=True ):
    with torch.no_grad():
        y_test = y_test.flatten() 
        assert y_test.shape[0] == predictions.shape[0]
        uncertainty = iqr(predictions,dim=-1) if quantile_uncertainty else predictions.std(dim=-1)
        proximity = torch.cdist( x_test.reshape(x_test.shape[0],-1), x_train.reshape(x_train.shape[0],-1) ).min(dim=-1).values
        assert proximity.shape == uncertainty.shape
        uncertainty  =  uncertainty.cpu().numpy()
        proximity    =    proximity.cpu().numpy()
        #
        # ~~~ Use polynomial regression to measure the strength of the relation uncertainty~proximity
        fits = [ univar_poly_fit( y=uncertainty, x=proximity, degree=k ) for k in (1,2,3,4) ]
        polys = [ fit[0] for fit in fits ]
        R_squared_coefficients = [ fit[2] for fit in fits ]
        if show:
            points_with_curves(
                    y = uncertainty,
                    x = proximity,
                    points_label = "Values on the Val./Test Set",
                    marker_color = "grey",
                    curves = polys,
                    curve_labels = [ f"R^2 {val:.3}" for val in R_squared_coefficients ],
                    title  =  "Uncertainty vs Proximity to Observed Data (with Polynomial Fits)",
                    xlabel = "Distance to the Nearest Training Data Point",
                    ylabel = "Uncertainty (viz. Posterior Predictive " + "IQR)" if quantile_uncertainty else "std. dev.)",
                    model_fit = False
                )
        slope_in_OLS = fits[0][1][0]
        R_squared_of_OLS = R_squared_coefficients[0]
        if (max(R_squared_coefficients) > R_squared_of_OLS+0.1):
            my_warn(f"The OLS fit (R^2 {R_squared_of_OLS:.4}) was outperformed by a higher degree polynomial fit (R^2 {max(R_squared_coefficients):.4}). Using either beta_1 from OLS or the correlation may result in an inaccurate quantification of the relation. Please check the plot returned by the `show=True` argument.")
        return slope_in_OLS, cor(uncertainty,proximity)


# the ones from the SLOSH paper


