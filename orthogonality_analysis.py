
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression

def orthogonality_analysis(y_actual, y_pred, df_predictors):
    residuals = y_actual - y_pred
    results = []

    for col in df_predictors.columns:
        x = df_predictors[[col]].dropna()
        r = residuals.loc[x.index]

        if x.shape[0] < 2 or r.shape[0] < 2:
            continue  # skip if insufficient data

        # Linear regression
        model = LinearRegression().fit(x, r)
        r2 = model.score(x, r)
        corr = np.corrcoef(x.squeeze(), r)[0, 1]

        # Mutual information (requires no NaNs)
        try:
            mi = mutual_info_regression(x, r, discrete_features=False)[0]
        except:
            mi = np.nan

        results.append({
            'predictor': col,
            'R_squared': r2,
            'correlation': corr,
            'mutual_info': mi
        })

    results_df = pd.DataFrame(results).sort_values(by='mutual_info', ascending=False)
    return results_df
