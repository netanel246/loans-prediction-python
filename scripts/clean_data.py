import pandas as pd


def consolidate_home_ownership_values(val):
    if val == "HaveMortgage":
        val.replace("HaveMortgage", "Home Mortgage")
    else:
        return val


# TODO - find a more generic way
def consolidate_purpose_values(val):
    if val == "other":
        return "Other"
    elif val == "major_purchase":
        return "Major Purchase"
    elif val == "small_business":
        return "Small Business"
    elif val == "renewable_energy":
        return "Renewable Energy"
    elif val == "wedding":
        return "Wedding"
    elif val == "vacation":
        return "Vacation"
    elif val == "moving":
        return "Moving"
    else:
        return val


def replace_col_space_with_underscore(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df


def replace_space_in_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].map(lambda v: str(v).replace(" ", "_"))
    return df


def replace_unsupported_chars(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.map(lambda col: col.replace(">", "gt")
                                               .replace("<", "lt")
                                               .replace("+", "plus"))
    return df


if __name__ == "__main__":
    train_data = pd.read_csv("../data/raw_bank_loan_status.csv")
    train_data_cleaned = replace_col_space_with_underscore(train_data)

    # Handle nan values in y and X
    train_data_cleaned.dropna(axis=0, subset=["Loan_Status"], how="any", inplace=True)

    # drop not useful features
    train_data_cleaned.drop(["Loan_ID", "Customer_ID", "Tax_Liens"], axis=1, inplace=True)

    consolidate_home_ownership = train_data_cleaned.Home_Ownership.apply(consolidate_home_ownership_values)
    train_data_cleaned["Home_Ownership"] = consolidate_home_ownership

    consolidate_purpose = train_data_cleaned.Purpose.apply(consolidate_purpose_values)
    train_data_cleaned["Purpose"] = consolidate_purpose

    train_data_cleaned = replace_space_in_data(train_data_cleaned)
    train_data_cleaned = replace_unsupported_chars(train_data_cleaned)

    train_data_cleaned.to_csv("../data/bank_loan_status.csv")
