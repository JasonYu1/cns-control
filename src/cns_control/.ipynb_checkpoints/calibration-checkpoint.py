import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
from tqdm.auto import tqdm

def min_max(arr, MIN=130, MAX=10000):
    arr[arr > MAX] = MAX
    arr[arr < MIN] = 0
    return arr
    
def hermite_gaussian(x, y, x0, y0, w0, A):
    # Calculate the Hermite-Gaussian TEM_11 or TEM_22 mode
    z = 0  # Assume propagation distance z = 0 for simplicity
    k = 2 * np.pi / 0.5  
    z_R = np.pi * w0**2 / k
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)

    # Shift the coordinates
    x_shifted = x - x0
    y_shifted = y - y0

    # Hermite polynomial
    # HG_1 = lambda xi: 2 * xi # order 1
    def HG_1(xi):
        return 2*xi
    # HG_2 = lambda xi: 4*xi**2 - 2 # order 2

    amplitude = (A / w_z) * HG_1(np.sqrt(2) * x_shifted / w_z) * HG_1(np.sqrt(2) * y_shifted / w_z)
    amplitude *= np.exp(-(x_shifted**2 + y_shifted**2) / w_z**2)

    # asym_factor1 = 1 + asym1 * (x_shifted + y_shifted) / np.sqrt(2 * (x_shifted**2 + y_shifted**2 + 1e-6))
    # amplitude *= asym_factor1

    # asym_factor2 = 1 + asym2 * (x_shifted - y_shifted) / np.sqrt(2 * (x_shifted**2 + y_shifted**2 + 1e-6))
    # amplitude *= asym_factor2

    # skew_factor = 1 + skew * (x_shifted * y_shifted) / np.sqrt(2 * (x_shifted**2 + y_shifted**2 + 1e-6))
    # amplitude *= skew_factor
    
    return np.abs(amplitude)
    # return amplitude

def fit_hermite_gaussian(data, initial_guess, bounds):
    y, x = np.indices(data.shape)
    x_flat = x.flatten()
    y_flat = y.flatten()
    data_flat = data.flatten()

    popt, _ = curve_fit(lambda coords, x0, y0, w0, A: hermite_gaussian(coords[0], coords[1], x0, y0, w0, A).flatten(),
                        (x_flat, y_flat), data_flat, p0=initial_guess, bounds=bounds, max_nfev=1e3)

    return popt


def fit_coms(ds, box=30, r=5):

    lower_bounds = [-box-r, -box-r, -np.inf, -np.inf]
    upper_bounds = [box+r, box+r, np.inf, np.inf]
    bounds = (lower_bounds, upper_bounds)
    initial_guess = [box, box, 15, 3000]  
                

    coms = []
    for img in tqdm(ds['imgs'].values):
        dat = img
        # dat = min_max(dat)
        thres = 1000

        if np.max(dat) > thres:
            mask = min_max(dat.copy()) > thres
            com = ndi.center_of_mass(mask)
            data = dat[int(com[0])-box:int(com[0])+box, int(com[1])-box:int(com[1])+box]
            
            try:
                params = fit_hermite_gaussian(data, initial_guess, bounds)
                coms.append((com[1]+params[0]-box, com[0]+params[1]-box))
            except:  # noqa: E722
                if np.max(dat) > 4000:
                    coms.append((com[1], com[0]))
                else:
                    coms.append((np.nan, np.nan))
            
            
        elif np.max(dat) > 300:
            mask = min_max(dat.copy()) > 300
            com = ndi.center_of_mass(mask)
            data = dat[int(com[0])-box:int(com[0])+box, int(com[1])-box:int(com[1])+box]
            
            try:
                params = fit_hermite_gaussian(data, initial_guess, bounds)
                coms.append((com[1]+params[0]-box, com[0]+params[1]-box))
            except:  # noqa: E722
                coms.append((np.nan, np.nan)) 
        
        else:
            coms.append((np.nan, np.nan))
            
    return np.asarray(coms)
    