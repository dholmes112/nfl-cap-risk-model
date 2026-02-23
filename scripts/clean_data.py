import pandas as pd

def clean_cap_data(df):

    df = df.copy()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Clean financial columns safely
    money_cols = ["Cap Hit", "Dead Cap", "Cash Total"]

    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.replace(r"\((.*?)\)", r"-\1", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean any percentage column dynamically (no hardcoding)
    pct_cols = [col for col in df.columns if "Pct" in col]

    for col in pct_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df