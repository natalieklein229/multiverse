
import torch
from matplotlib import pyplot as plt
from bnns.utils import iqr, univar_poly_fit
from quality_of_life.my_visualization_utils import points_with_curves

#
# ~~~ Measure MSE of the predictive median relative 
def mse_of_median( predictions, y_test ):
    with torch.no_grad():
        y_test = y_test.flatten()
        pred = predictions.median(dim=-1)
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
#         return (( pred - y_test )**2).mean(dim=0).median()

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
        accuracy = (predictions.median(dim=-1)-y_test)**2 if quantile_accuracy else (predictions.median(dim=-1)-y_test)**2
        uncertainty  =  uncertainty.cpu().numpy()
        accuracy     =     accuracy.cpu().numpy()
        #
        # ~~~ Use polynomial regression to measure the strength of the relation uncertainty~accuracy
        fits = [ univar_poly_fit( y=uncertainty, x=accuracy, degree=k ) for k in (1,2,3,4) ]
        polys = [ fit[0] for fit in fits ]
        if show:
            points_with_curves( y=uncertainty, x=accuracy, curves=polys, title="Uncertainty vs Accuracy (with Polynomial Fits)", xlabel="Accuracy", ylabel="Uncertainty", model_fit=False, show=True )
        R_squared_coefficients = [ fit[0] for fit in fits ]
        return R_squared_coefficients

#
# ~~~ Measure strength of the relation "predictive uncertainty (std. dev.)" ~ "distance from training points"
def uncertainty_vs_distance_to_data( predictions, y_test, quantile_uncertainty, x_test, x_train, show=True ):
    with torch.no_grad():
        y_test = y_test.flatten() 
        assert y_test.shape[0] == predictions.shape[0]
        uncertainty = iqr(predictions,dim=-1) if quantile_uncertainty else predictions.std(dim=-1)
        proximity = torch.cdist(x_test,x_train).min(dim=-1).values
        assert proximity.shape == uncertainty.shape
        uncertainty  =  uncertainty.cpu().numpy()
        proximity    =    proximity.cpu().numpy()
        #
        # ~~~ Use polynomial regression to measure the strength of the relation uncertainty~proximity
        fits = [ univar_poly_fit( y=uncertainty, x=proximity, degree=k ) for k in (1,2,3,4) ]
        polys = [ fit[0] for fit in fits ]
        if show:
            points_with_curves( y=uncertainty, x=proximity, curves=polys, title="Uncertainty vs Accuracy (with Polynomial Fits)", xlabel="Accuracy", ylabel="Uncertainty" )
        R_squared_coefficients = [ fit[0] for fit in fits ]
        return R_squared_coefficients


# the ones from the SLOSH paper


