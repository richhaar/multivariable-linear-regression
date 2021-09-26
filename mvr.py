import numpy as np
import matplotlib.pyplot as plt
import itertools

import mvr_calc

class MultiVariableRegression:
    """
    Class to manage the multivariable regression, holding a data class,
    which has it's own calculation class to handle regression calculations.
    """
        
    class Data:
        """Holds the raw numerical data for the regression calculations"""
        def sanityCheck(self):
            """
            Test statistics rely on the degrees of freedom being
            greater than 0, Having more data than the number of
            variable predictors (x.. x_n) by atleast 2 will satisfy
            the sanity check.
            """
            if self.df > 0:
                self.sane = True
                
        def populateMetaData(self):
            self.ndim  = self.c.getMatrixWidth(self.x)
            self.ndata = self.y.size
            self.df    = self.ndata - self.ndim - 1

            self.x_matrix = self.c.addOnesToData(self.x, self.ndata, self.ndim)

        def populateVariance(self):
            self.y_bar = self.c.calcAverage(self.y)
            self.x_bar = self.c.calcAverage(self.x)

            self.y_variance = self.c.calcVariance(self.y,self.y_bar)
            self.x_variance = self.c.calcVariance(self.x,self.x_bar).reshape(
                self.ndata, self.ndim)

            self.y_variance_sq = self.c.calcSumProduct(
                self.y_variance,self.y_variance)
            
            self.x_variance_sq = np.zeros(self.ndim)
            self.x_y_variance  = np.zeros(self.ndim)

            for n in range(0, self.ndim):
                x_var = self.x_variance[:,n]
                self.x_variance_sq[n] = self.c.calcSumProduct(
                    x_var, x_var)
                self.x_y_variance[n]  = self.c.calcSumProduct(
                    x_var, self.y_variance)

        def populateCorrelation(self):
            self.correlation = self.c.calcCorrelation(
                self.ndim,
                self.x_y_variance,
                self.x_variance_sq,
                self.y_variance_sq)

        def populateRegression(self):
            self.s_matrix = self.c.findSMatrix(self.x_matrix)
            self.regression = self.c.calcRegression(
                self.s_matrix, self.x_matrix, self.y)

        def populateEstimationData(self):
            self.y_hat   = np.dot(self.x_matrix, self.regression)
            self.y_error = self.y - self.y_hat
            
            self.sum_errors_sq = self.c.calcSumProduct(
                self.y_error, self.y_error)
            
            self.adjusted_R_sq = self.c.findAdjustedRSquared(
                self.sum_errors_sq, self.y_variance_sq, self.ndata, self.df)
            
        def __init__(self,x,y,c):
            self.x = x
            self.y = y
            self.c = c
            self.populateMetaData()
            self.sanityCheck()
            if self.sane:
                self.populateVariance()
                self.populateCorrelation()
                self.populateRegression()
                self.populateEstimationData()
            
    def ANOVA(self,core_data,significance):
        """
        Run the analysis of variance test on the data. The analysis
        is run with a null hypothesis to check whether the population
        regression coefficients (A_0 ... A_n) are 0. Such that the data
        follows a normal distribution with mean B and standard deviation
        sigma. By rejecting the hypothesis, we can say with the given
        significance the x dimensions do affect the y data. And
        uses the F distribution to determine the critical value.
        
        Run once for All A_0,... A_n being 0, which can be rejected
        if any A is not 0. Then run for each A_x.
        """
        print(f'Running Analysis of Variance with given significance:'
              + f'{significance}\nNull hypothesis: A_0...A_n are all 0')
        
        test_statistic = self.c.calcTestStatisticAllX(
            core_data.y_variance_sq,
            core_data.sum_errors_sq,
            core_data.ndim,
            core_data.df)

        critical_value = core_data.c.findCriticalFValue(
            core_data.ndim,
            core_data.df,
            significance)

        if critical_value < test_statistic:
            print(f'Null hypotehesis rejected '
                  + f'{critical_value:.3} < {test_statistic:.3}\n'
                  + f'Atleast one A_x is not 0 with certainty {significance}')
        else:
            print(f'Null hpothesis accepted'
                  + f'{critical_value:.3} > {test_statistic:.3}\n'
                  + f'All A_x are 0 with certainty {significance}')
            
        if core_data.ndim > 1:
            for n in range(0,core_data.ndim):
                test_statistic = self.c.calcTestStatisticSingleX(
                    core_data.regression,
                    core_data.s_matrix,
                    core_data.sum_errors_sq,
                    n,
                    core_data.df)
                
                critical_value = core_data.c.findCriticalFValue(
                    core_data.ndim,
                    core_data.df,
                    significance)

                if critical_value < test_statistic:
                    print(f'Null hypotehesis rejected '
                          + f'{critical_value:.3} < {test_statistic:.3}\n'
                          + f'A_{n} is not 0 with certainty {significance}')
                else:
                    print(f'Null hpothesis accepted'
                          + f'{critical_value:.3} > {test_statistic:.3}\n'
                          + f'A_{n} is 0 with certainty {significance}')


    def roundRobin(self):
        """
        Calculate the adjusted R^2 value for all combinations of
        predictor variables, to determine which predictor variables
        are best suited to the regression.
        """
        # Starting with 1 predictor variable up to n variables
        for n in range(0, self.core_data.ndim):
            # Generate all combinations of predictor variables
            for i in itertools.combinations(range(0, self.core_data.ndim), n+1):
                x = self.core_data.x[:,i]
                sub_data = self.Data(x, self.core_data.y, self.c)
                
                print(f'For predictor variables: {i}'
                      + f': regression = {sub_data.regression}'
                      + f'R^2 = {sub_data.adjusted_R_sq}')


    def plotData(self):
        """
        Plot each predictor variable against y, showing the correlation
        co-efficient for each variable. Additionally plots the regression
        line if there is only one predictor variable.

        """
        for n in range(0,self.core_data.ndim):
            #plt.subplot(self.core_data.ndim, 1, n+1)
            plt.figure(n)
            x = self.core_data.x[:,n]
            y = self.core_data.y
            plt.plot(x,y,'o')

            plt.title(f'Data correlation of:'
                      + f'{self.core_data.correlation[n]:.4}',
                      fontsize=20)
            plt.ylabel(f'{self.headers[-1]}',fontsize=18)
            plt.xlabel(f'{self.headers[n]}',fontsize=18)

            
        # Regression line only makes sense to plot with 1 predictor variable   
        if self.core_data.ndim == 1:
            a = self.core_data.regression[[n,self.core_data.ndim]]
            x_matrix= self.c.addOnesToData(x, x.size, 1)
            y_hat = np.dot(x_matrix, a)
            plt.plot(x, y_hat, '-',label='Regression line')
            plt.legend()

        plt.show()
        
    def estimateData(self):
        """
        For a given input list of data, provide an estimate y value,
        along with a
        confidence interval: (the interval of the mean y value),
        prediction_interval: (the interval of predicted values).
        """
        if self.estimates is not None:
            n = self.estimates.size // self.core_data.ndim
            x = self.c.addOnesToData(self.estimates, n, self.core_data.ndim)

            y_hat = np.dot(x, self.core_data.regression)

            fval = self.c.findCriticalFValue(
                1, self.core_data.df, self.confidence)

            for i in range(0,n):
                x_n = x[i,:-1]

                mahalanobis_distance = self.c.getMahalanobisDistance(x_n,
                    self.core_data.x_bar,
                    self.core_data.ndim,
                    self.core_data.ndata,
                    self.core_data.s_matrix)

                # fval, sum_errors_sq, df, ndata, mahalanobis_distance):
                confidence_interval = self.c.getConfidenceInterval(
                    self.core_data.sum_errors_sq,
                    self.core_data.df,
                    self.core_data.ndata,
                    mahalanobis_distance,
                    fval)
                
                prediction_interval = self.c.getPredictionInterval(
                    self.core_data.sum_errors_sq,
                    self.core_data.df,
                    self.core_data.ndata,
                    mahalanobis_distance,
                    fval)

                # TODO: add table formatting before loop
                print(f'x_n {x_n} y_hat {y_hat[i]:.3}'
                      + f' ci {confidence_interval[0,0]:.3}'
                      + f' pi {prediction_interval[0,0]:.3}'
                      + f' with confidence {self.confidence:.3}')

                
    def populateData(self):
        if not self.data_populated:
            self.core_data = self.Data(self.data[:,0:-1],self.data[:,-1],self.c)
            data_populated = True

    def runAnalysisOfVariance(self, significance):
         self.ANOVA(self.core_data,significance)
        
    def __init__(self, data, headers, confidence=0.95, estimates=None):
        self.data_populated = False
        self.data = data
        self.confidence = confidence
        self.c = mvr_calc.MVRCalculator()
        self.headers = headers
        self.estimates = estimates
        
        self.populateData()
        
        self.runAnalysisOfVariance(self.confidence)
        self.roundRobin()
        self.estimateData()
