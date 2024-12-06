from utils import setup_anlaysis_df

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.sparse import spdiags


# SECTION 1: Read and Merge Data
merged_df = setup_anlaysis_df()


# SECTION 2: Data Analysis
def get_ols_results(X, y):
  return sm.OLS(y, X).fit()

def get_weighted_ols_results(X, y, W):
  return sm.WLS(y, X, W).fit()

# analysis 0- unweighted model, no processing
N = merged_df.shape[0]

# explanatory variables are the distance to a drop box and bias
X = np.stack((merged_df['dist_to_nearest_ballot_box_km'], np.ones((N, ))), axis=-1)
# the response variable is the proportion of voters who voted
y = merged_df['prop_voted']

results_ols = get_ols_results(X, y)
print('\n\nAnalysis 0 Results')
print(results_ols.summary())

# analysis 1- weighted model, no processing
# weight districts by their count of eligible voters
W = spdiags(merged_df['total_voting_age_pop'], [0], N, N).toarray()

results_weighted_ols = get_weighted_ols_results(X, y, np.diag(W))
print('\n\nAnalysis 1 Results')
print(results_weighted_ols.summary())

# analysis 2- unweighted model, cap proportions to 1
y_cap = merged_df['prop_voted'].where(merged_df['prop_voted'] < 1., 1.)
print('Max y cap', max(y_cap))

results_ols = get_ols_results(X, y_cap)
print('\n\nAnalysis 2 Results')
print(results_ols.summary())

# analysis 3- cap proportions to 1
# the response variable is the proportion of voters who voted, capped to 1
results_weighted_ols = get_weighted_ols_results(X, y_cap, np.diag(W))
print('\n\nAnalysis 3 Results')
print(results_weighted_ols.summary())
