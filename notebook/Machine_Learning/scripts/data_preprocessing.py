import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    # Encoding categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Scaling numerical features
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

    return df, scaler, label_encoders

if __name__ == "__main__":
    df = pd.read_csv('../data/bank_marketing.csv')
    df_preprocessed, scaler, label_encoders = preprocess_data(df)
    df_preprocessed.to_csv('../data/bank_marketing_preprocessed.csv', index=False)
