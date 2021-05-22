import numpy as np
import pandas as pd

def csv_to_matrix(sigma_path, mu_path, date = '2021-03-01', assets_list = []):

    """
    Retrieves the covariance matrix sigma and the expectations vector mu
    from .csv files.
    """

    if not assets_list:

        complete_monthly_returns = pd.read_csv(mu_path)
        complete_monthly_returns.set_index('Date', inplace = True)

        cov = pd.read_csv(sigma_path)
        cov.set_index('Unnamed: 0', inplace = True)
        index = cov.index.to_list()

        mu = np.expand_dims(complete_monthly_returns.loc[date].to_numpy(),1)
        sigma = cov.to_numpy()

        return sigma, mu, index

    else:

        complete_monthly_returns = pd.read_csv(mu_path)
        complete_monthly_returns.set_index('Date', inplace = True)

        cov = pd.read_csv(sigma_path)
        cov.set_index('Unnamed: 0', inplace = True)
        index = cov.index.to_list()

        mu = np.expand_dims(complete_monthly_returns[assets_list].loc[date].to_numpy(),1)
        sigma = cov[assets_list].loc[assets_list].to_numpy()

        return sigma, mu, assets_list