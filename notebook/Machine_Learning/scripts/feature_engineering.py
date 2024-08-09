import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    return principal_components, pca

if __name__ == "__main__":
    df = pd.read_csv('../data/bank_marketing_preprocessed.csv')
    principal_components, pca = apply_pca(df.drop(columns=['y']))  # Exclude target variable
    pd.DataFrame(principal_components).to_csv('../data/bank_marketing_pca.csv', index=False)
