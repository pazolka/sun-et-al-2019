import numpy as np

# Define the custom metrics
def custom_nse(grace, gldas, y_pred):
    spatial_mean_per_ex_grace = grace.mean(dim=['lat', 'lon'], skipna=True)
    spatial_mean_per_ex_gldas = gldas.mean(dim=['lat', 'lon'], skipna=True)
    spatial_mean_per_ex_pred = y_pred.mean(dim=['lat', 'lon'], skipna=True)
    numerator = ((spatial_mean_per_ex_grace - spatial_mean_per_ex_gldas + spatial_mean_per_ex_pred)**2).sum()
    denominator = ((spatial_mean_per_ex_grace - spatial_mean_per_ex_grace.mean())**2).sum()
    return (1 - numerator / (denominator + 1e-07)).values

def custom_corr(grace, gldas, y_pred):
    # x is GRACE TWSA
    x = grace
    # y is CNN-corrected TWSA
    y = gldas - y_pred
    mx = x.mean()
    my = y.mean()
    x_m, y_m = x - mx, y - my
    r_num = (x_m * y_m).sum()
    x_square_sum = (x_m * x_m).sum()
    y_square_sum = (y_m * y_m).sum()
    r_den = np.sqrt(x_square_sum * y_square_sum)
    r = (r_num / (r_den + 1e-07)).values
    return r
