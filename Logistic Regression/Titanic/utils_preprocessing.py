import pandas as pd

def drop_columns(df, columns):
    return df.drop(columns=columns, axis=1)

def clean_data(df, verbose=False):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Not dataframe detected!")

    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    if verbose:
                        print(f"[INFO] Coloana numerica: {col} → completare cu media")
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    if verbose:
                        print(f"[INFO] Coloana categorica: {col} → completare cu moda")
                    df[col] = df[col].fillna(df[col].mode()[0])
    return df
