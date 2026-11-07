"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
def load_housing():
    url = 'https://github.com/ageron/data/raw/main/housing.tgz'
    path = os.path.join(tempfile.gettempdir(), "housing.tgz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    
    with tarfile.open(path) as housing_tar:
        csv_file = housing_tar.extractfile("housing/housing.csv")
        df = pd.read_csv(csv_file)
        return df
    return df

# Step 2 - income_categories
def income_categories(df):
    feature = pd.cut(df['median_income'], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],labels=[1, 2, 3, 4, 5]).astype(int)
    return feature #series is returned not dataframe

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split
def stratified_split(df, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state, stratify=income_categories(df))
    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    corr_matrix = df.corr(numeric_only=True)
    corr_matrix = corr_matrix["median_house_value"].drop("median_house_value").sort_values(ascending=False)
    return corr_matrix

# Step 5 - add_ratio_features
def add_ratio_features(df):
    df_copy = df.copy()
    df_copy['rooms_per_house'] = df['total_rooms']/df['households']
    df_copy['bedrooms_ratio'] = df['total_bedrooms']/df['total_rooms']
    df_copy['people_per_house'] = df['population']/df['households']
    return df_copy

# Step 6 - split_features_labels
def split_features_labels(df):
    features = df.drop('median_house_value', axis=1)
    labels = df["median_house_value"].copy()
    return features,labels

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init = 10,
            random_state = self.random_state
        )
        self.kmeans_.fit(X,sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Step 8 - numeric_pipeline (not yet solved)
# TODO: implement

# Step 9 - categorical_pipeline (not yet solved)
# TODO: implement

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

