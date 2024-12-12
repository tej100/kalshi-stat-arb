import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import matplotlib.pyplot as plt

class RNDPricer:
    def __init__(self, chain):
        # init df
        self.df = chain.copy()
        self.S0 = self.df['underlying'].values[0]

        # get OTM call iv and OTM put iv
        self.df = self.df[((self.df['option_type']=='c') & (self.df['strike'] >= self.S0)) | ((self.df['option_type']=='p') & (self.df['strike'] < self.S0))]
        
        self.df.sort_values('strike', inplace=True)
        self.df.dropna(subset=['Implied Volatility'], inplace=True)
        
        # check if chain is empty
        self.warning_flag = False
        if self.df.empty:
            raise ValueError('Empty chain')
        
        # init constants
        self.F = self.df['fwd'].values[0]
        self.T = self.df['T'].values[0] / 365
        self.DF = self.df['DF'].values[0]
        self.day = self.df['quote'].values[0]
        
        # init variables
        self.strikes = self.df['strike'].values
        self.iv = self.df['Implied Volatility'].values
        
        # interpolation range over strike
        # necessary evil to make everything work
        self.K_grid = np.linspace(max(self.strikes.min()-2000, 100), self.strikes.max()+1000, 5000)
        
        # init RND for pricing
        self.__init_rnd()
        
        # check warning flag
        if self.warning_flag:
            print(f"Warning: potentially noisy IV curve on {self.day}")
        
    def __init_rnd(self):
        spline, iter = self.__fit_spline_without_warnings()
        
        # smoothing warning for extremely noisy days
        # could lead to bad pricing
        if iter > 2:
            self.warning_flag = True
        
        # smooth vols and then call prices
        smooth_vols = spline(self.K_grid) / 100
        smooth_calls = self.__bs_call_surface(smooth_vols)
        
        # get 2nd derivative of call prices
        fp = np.gradient(smooth_calls, self.K_grid)
        fpp = np.gradient(fp, self.K_grid)
        
        # trim ends by two for gradient errors
        fpp = fpp[2:-2]
        self.K_grid = self.K_grid[2:-2]
        
        # lhs, enforce decreasing
        peak = np.argmax(fpp)        
        if fpp[0] > 0:
            lhs_min = np.argmin(fpp[:peak])
            fpp[:lhs_min] = np.flip(np.minimum.accumulate(np.flip(fpp[:lhs_min])))
            
        # rhs, enforce decreasing 
        if fpp[-1] > 0:
            rhs_min = peak + np.argmin(fpp[peak:])
            fpp[rhs_min:] = np.minimum.accumulate(fpp[rhs_min:])
        
        # rhs, correct for positive values with linear interp to zero
        if fpp[-1] > 0:
            rhs_min = peak + np.argmin(fpp[peak:])
            
            # linear interp to zero
            fpp[rhs_min:] = np.interp(self.K_grid[rhs_min:], [self.K_grid[rhs_min], self.K_grid[-1]], [fpp[rhs_min], 0.0])
        
        
        # enforce global bound by zero
        fpp = np.maximum(fpp, 0)
        
        # normalizing to make integral of cdf = 0
        area = np.trapz(fpp, self.K_grid)
        self.rnd = fpp / area
        
    def __fit_spline_without_warnings(self, maxiter=10):
        # set weights for interpolation to be less important in tails
        weights = np.ones_like(self.strikes)
        #weights[np.abs(self.strikes - self.S0) > 1000] = 0.25
        
        # initialization as per scipy.interpolate.UnivariateSpline
        s = len(weights)
        attempt = 0
        while attempt < maxiter:
            #print(f"Attempt {attempt + 1} with s = {s}")
            with warnings.catch_warnings(record=True) as w_records:
                warnings.simplefilter("always")

                # 4th order chosen as in literature
                spline = UnivariateSpline(self.strikes, self.iv, w=weights, k=4, s=s)
                
                # if we have a warning we double smoothing factor
                if any("maxit" in str(warn.message) for warn in w_records):
                    s *= 2
                    attempt += 1
                else:
                    return spline, attempt
        
        print("Warning: no ideal smoothing for vol curve")
        return spline, attempt
        
        
    def __bs_call_surface(self, smooth_vols):
        d1 = (np.log(self.F / self.K_grid) + (0.5 * smooth_vols ** 2) * self.T) / (smooth_vols * np.sqrt(self.T))
        d2 = d1 - smooth_vols * np.sqrt(self.T)
        return self.DF * (self.F * norm.cdf(d1) - self.K_grid * norm.cdf(d2))
        
    def get_rnd(self):
        return pd.DataFrame({'strike': self.K_grid, 'density': self.rnd})
    
    def price_bucket(self, lower, upper):
        # get index mask
        mask = np.logical_and(self.K_grid >= lower, self.K_grid <= upper)
        
        # if no strikes in interval we can price at 0.00
        if mask.sum() == 0:
            return 0.00
        
        # integrate RND over interval
        target_densities = self.rnd[mask]
        target_strikes = self.K_grid[mask]
        return np.trapz(target_densities, target_strikes)
    
    def price_tail(self, strike, lower_tail=True):
        # get index mask
        mask = self.K_grid < strike if lower_tail else self.K_grid > strike
        
        # throw error
        if mask.sum() == 0:
            raise ValueError('No strikes in interval')
        
        # integrate RND over interval
        target_densities = self.rnd[mask]
        target_strikes = self.K_grid[mask]
        return np.trapz(target_densities, target_strikes)
    
    
        
        
        
        
        
        
    