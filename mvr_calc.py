import numpy as np
from scipy.stats import f

class MVRCalculator:
    """
    Class holds the calculations needed to perform the regression
    on some data. Used to seperate out the data and calculations.
    """
    
    @staticmethod
    def searchValue(f, target,
                    tolerance=0.000001, start=0, step_size=1, damping=0.5):
        """
        Finds x for a given target y, for a given linear function f(x).
        Works iteratively through values of x to find the target f(x)
        value, once the target is 'found', the step gets reversed
        and damped until the target is found within the given tolerance.
        """
        def stepDirection(increasing, lower):
            """
            Finds whether x should increase of decrease,
            depending if the f(x) function is an increasing or decreasing
            function and if f(x_0) is lower than f(x_target)
            """
            if (increasing and lower) or (not increasing and not lower):
                return  1
            else:
                return -1

        x,error,a0,a1 = start, tolerance+1, f(start), f(start+step_size)
        increasing, start_lower = a1 > a0, a0 < target

        step_direction = stepDirection(increasing, start_lower)
        step = step_direction * step_size

        while abs(error) > tolerance :
            x = x + step   
            a = f(x)

            error = target - a
            lower = error > 0

            new_step_direction = stepDirection(increasing, lower)

            # If true, the target x is between f(x) and f(x-step)  
            if step_direction != new_step_direction:
                step_size = damping * step_size

            step = new_step_direction * step_size                
        return x

    @staticmethod
    def addOnesToData(x,ndata,ndim):
        """Adds a column of 1s to a given input vector or matrix"""
        #if len(x.shape) == 1:
        #    x = np.expand_dims(x, axis=0)
        x = x.reshape(ndata,ndim)
        return np.append(x,np.ones((ndata,1)), axis=1)

    @staticmethod
    def calcSumProduct(vector1,vector2):
        """Returns the sum of the product of two vectors"""
        return np.sum(vector1 * vector2)

    @staticmethod
    def calcCorrelation(ndim, x_y_variance, x_variance_sq, y_variance_sq):
        """
        Calculates the correlation between x and y data
        for each x dimension
        """
        coefficients = np.zeros(ndim)
        for n in range(0,ndim):
            coefficients[n] = x_y_variance[n] / np.sqrt(
                x_variance_sq[n] * y_variance_sq)
            
        return coefficients

    @staticmethod
    def calcRegression(s_matrix,x_matrix,y):
        """Calculates the regression equation (a_0 -> a_n + b)"""
        return np.dot(s_matrix, np.dot(x_matrix.T, y))

    @staticmethod
    def findSMatrix(x_matrix):
        return np.linalg.inv(np.dot(x_matrix.T,x_matrix))

    @staticmethod
    def findAdjustedRSquared(sum_errors_sq,y_variance_sq,ndata,df):
        """
        Finds R^2, adjusted for the fact that normally R^2 will
        increase for added predictor variables regardless if the variable
        is a good predictor or not.
        """
        return  1 - ((sum_errors_sq / df) / (y_variance_sq / (ndata - 1)))

    @staticmethod
    def getMahalanobisDistance(x_n, x_bar, ndim, ndata, s_matrix):
        """Get the mahalanobis distance of a given x_n"""
        x = (x_n - x_bar).reshape(ndim,1)
        return np.dot(x.T,np.dot(s_matrix[:-1,:-1],x)) * (ndata - 1)

    @staticmethod
    def findCriticalFValue(ndim, df, significance):
        """
        Find F distribution values, used as critical values in
        Analysis of variance tests.
        """
        return MVRCalculator.searchValue(lambda z: f.cdf(z,ndim,df),
                                            significance)

    @staticmethod
    def getConfidenceInterval(
            sum_errors_sq, df, ndata, mahalanobis_distance, fval):
        """
        Interval range for the mean value of a predicted y, to account
        for the variance in the population data. With the confidence
        (e.g. 0.95) determined by fval.
        """
        return np.sqrt(fval
                       * (1/ndata + mahalanobis_distance / (ndata -1))
                       * (sum_errors_sq / df))

    @staticmethod
    def getPredictionInterval(
            sum_errors_sq, df, ndata, mahalanobis_distance, fval):
        """
        Interval range to give a probable range of future values.
        This range will be higher than the confidence interval,
        to account for the fact that the mean predicted value
        can vary by the confidence value, and then additionally
        the value can vary from that mean.
        """
        return np.sqrt(fval
                       * (1 + 1/ndata + mahalanobis_distance / (ndata - 1))
                       * (sum_errors_sq / df))

    @staticmethod
    def getMatrixWidth(v):
        """Function to find the width of a given numpy vector or matrix"""
        if len(np.shape(v)) > 1:
            return np.shape(v)[1]
        else:
            return 1

    @staticmethod
    def autoCorrelationTest(y_error, sum_errors,sq):
        """
        Check for auto correlation in our y data using the
        Durbin-Watson statistic, a result lower than 1
        may indicate the presence of autocorrelation.
        """
        residual = y_error[1:] - y_error[:-1]
        return (MVRCalculator.calcSumProduct(residual, residual)
                / sum_errors_sq)

    @staticmethod
    def calcAverage(m):
        return np.mean(m,axis=0)

    @staticmethod
    def calcVariance(v,v_bar):
        return v - v_bar
    
    @staticmethod
    def calcTestStatisticAllX(y_variance_sq,sum_errors_sq,ndim,df):
        """
        Calculate the test statistic for the analysis of variance
        where the Null hypothesis is that the population A_1 -> A_n
        are all equal to 0. Such that the null hypothesis gets
        rejected if any A_x != 0.
        """
        return (((y_variance_sq - sum_errors_sq) / ndim)
                / (sum_errors_sq / df))

    @staticmethod
    def calcTestStatisticSingleX(regression, s_matrix, sum_errors_sq, n, df):
        """
        Calculate the test statistic for the analysis of variance
        where the Null hypothesis is that the population A_n is 0.
        Such that the null hypothesis gets rejected if A_n != 0.
        """
        return (regression[n]**2 / s_matrix[n,n]) / (sum_errors_sq / df)
