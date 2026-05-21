import pandas as pd

def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['combined_text'] = (
        df.get('NAME', '').fillna('') + " " +
        df.get('CONTAINS', '').fillna('') + " " +
        df.get('USES', '').fillna('') + " " +
        df.get('BENEFITS', '').fillna('') + " " +
        df.get('SIDE_EFFECT', '').fillna('')
    ).str.lower()
    return df