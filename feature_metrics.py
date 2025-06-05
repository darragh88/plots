import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

def compute_feature_metrics(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Given a DataFrame `df` and the name of a numeric target column `target_col`, compute:
      - Pearson correlation of each feature with the target
      - Mutual information between each feature and the target
      - LightGBM feature importances (split and gain)
    Returns a DataFrame indexed by feature name, with columns:
      'dtype', 'pearson_corr', 'mutual_info', 'lgb_split_importance', 'lgb_gain_importance'.
    """

    # 1) Separate features (X) and target (y)
    X_full = df.drop(columns=[target_col])
    y = df[target_col].values
    feature_names = X_full.columns.tolist()

    # Record original dtypes
    dtypes = X_full.dtypes.to_dict()

    # 2) Identify categorical vs. numerical columns
    #    We consider 'object' and 'category' dtypes as categorical.
    cat_cols = [col for col in feature_names if pd.api.types.is_categorical_dtype(df[col]) 
                or df[col].dtype == object]
    num_cols = [col for col in feature_names if col not in cat_cols]

    # 3) Prepare a copy of X where all features are numeric:
    #    - numeric columns → float64
    #    - categorical columns → integer codes (LabelEncoder)
    X_encoded = pd.DataFrame(index=df.index)

    # 3a) Numeric features: cast to float
    for col in num_cols:
        X_encoded[col] = df[col].astype(float)

    # 3b) Categorical features: label‐encode to integer codes
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # If there are NaNs, we fill them with a new category first:
        col_vals = df[col].astype("category").copy()
        if col_vals.isnull().any():
            col_vals = col_vals.cat.add_categories("__nan__").fillna("__nan__")
        X_encoded[col] = le.fit_transform(col_vals.astype(str))
        le_dict[col] = le

    # 4) Compute Pearson correlation between each (encoded) feature and the target
    pearson_corr = {}
    for col in feature_names:
        # Use the encoded‐or‐float array to compute correlation
        corr = np.corrcoef(X_encoded[col].values, y)[0, 1]
        pearson_corr[col] = corr

    # 5) Compute mutual information (MI) between each feature and the target
    #    mutual_info_regression can accept a 2D array X_encoded and list of discrete_features
    #    where discrete_features[i] = True if that column is discrete/categorical.
    X_arr = X_encoded.values
    # Build discrete_features mask: True for categorical columns
    discrete_mask = np.array([col in cat_cols for col in feature_names])
    mi_vals = mutual_info_regression(X_arr, y, discrete_features=discrete_mask, random_state=0)
    mutual_info = dict(zip(feature_names, mi_vals))

    # 6) Train a LightGBM regressor to get feature importances
    #    We want to pass the original DataFrame so that LightGBM can handle categories internally.
    #    First, ensure that categorical columns are dtype "category"
    X_lgb = X_full.copy()
    for col in cat_cols:
        X_lgb[col] = X_lgb[col].astype("category")

    # Construct LightGBM dataset
    lgb_dataset = lgb.Dataset(
        data=X_lgb,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    # Set up simple params; you can tweak learning_rate / num_leaves etc.
    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        # You may want to fix a random seed for reproducibility:
        "seed": 0,
    }

    # Train a small model (e.g., 100 boosting rounds)
    lgb_model = lgb.train(
        params=params,
        train_set=lgb_dataset,
        num_boost_round=100,
        verbose_eval=False,
    )

    # Extract importances
    gain_importances = lgb_model.feature_importance(importance_type="gain")
    split_importances = lgb_model.feature_importance(importance_type="split")

    lgb_gain = dict(zip(feature_names, gain_importances))
    lgb_split = dict(zip(feature_names, split_importances))

    # 7) Assemble everything into a single DataFrame
    results = pd.DataFrame(index=feature_names)
    results["dtype"] = [str(dtypes[col]) for col in feature_names]
    results["pearson_corr"] = [pearson_corr[col] for col in feature_names]
    results["mutual_info"] = [mutual_info[col] for col in feature_names]
    results["lgb_split_importance"] = [lgb_split[col] for col in feature_names]
    results["lgb_gain_importance"] = [lgb_gain[col] for col in feature_names]

    return results

# If run as a script, you could add example usage here:
if __name__ == "__main__":
    import sys
    import pickle

    # Example: python feature_metrics.py data.csv target_column
    if len(sys.argv) != 3:
        print("Usage: python feature_metrics.py <path_to_csv> <target_column>")
        sys.exit(1)

    csv_path = sys.argv[1]
    target_column = sys.argv[2]

    df = pd.read_csv(csv_path)
    metrics_df = compute_feature_metrics(df, target_col=target_column)
    metrics_df.to_csv("feature_metrics_output.csv", index=True)
    print("Feature metrics saved to feature_metrics_output.csv")
