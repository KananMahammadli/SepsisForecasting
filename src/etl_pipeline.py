import numpy as np


def preprocess_data(
    df,
    patient_id_col,
    temporary_target_col,
    n_min_rows_threshold,
    vital_signs,
    lab_features,
    output_length,
):
    # keep only patients who doesn't have sepsis at its first row
    df_filtered = (
        df.groupby(patient_id_col)
        .filter(lambda x: x[temporary_target_col].iloc[0] == 0)
        .reset_index(drop=True)
    )

    # keep only patients who has at least n_min_rows_threshold rows
    df_filtered_enough = (
        df_filtered.groupby(patient_id_col)
        .filter(lambda x: x.shape[0] >= n_min_rows_threshold)
        .reset_index(drop=True)
    )

    # for lab features, fill missing values with 0, and non-missing values with 1
    for lab_feature in lab_features:
        df_filtered_enough[f"{lab_feature}_binary"] = df_filtered_enough[lab_feature].apply(
            lambda x: 0 if np.isnan(x) else 1
        )
        df_filtered_enough[lab_feature] = df_filtered_enough[lab_feature].apply(
            lambda x: 0 if np.isnan(x) else x
        )

    df_filled = (
        df_filtered_enough.groupby(by=[patient_id_col])
        .apply(lambda x: x.ffill().bfill())
        .reset_index(drop=True)
    )

    # filter patients who has nan values in any of the features
    df_filled = (
        df_filled.groupby(patient_id_col)
        .filter(lambda x: x.isnull().sum().sum() == 0)
        .reset_index(drop=True)
    )

    # for each patient add lag features
    df_filled = (
        df_filled.groupby(patient_id_col)
        .apply(
            lambda x: x.assign(
                **{
                    f"{col}_lag_{i}": x[col].shift(i)
                    for i in range(output_length, 13)
                    for col in vital_signs
                }
            )
        )
        .reset_index(drop=True)
    )

    # for each patient add rolling mean features
    df_filled = (
        df_filled.groupby(patient_id_col)
        .apply(
            lambda x: x.assign(
                **{
                    f"{col}_rolling_mean_{i}": x[col].shift(output_length).rolling(i).mean()
                    for i in [output_length, 12, 24]
                    for col in vital_signs
                }
            )
        )
        .reset_index(drop=True)
    )
    df_filled = (
        df_filled.groupby(patient_id_col)
        .apply(
            lambda x: x.assign(
                **{
                    f"{col}_rolling_min_{i}": x[col].shift(output_length).rolling(i).min()
                    for i in [output_length, 12, 24]
                    for col in vital_signs
                }
            )
        )
        .reset_index(drop=True)
    )
    df_filled = (
        df_filled.groupby(patient_id_col)
        .apply(
            lambda x: x.assign(
                **{
                    f"{col}_rolling_max_{i}": x[col].shift(output_length).rolling(i).max()
                    for i in [output_length, 12, 24]
                    for col in vital_signs
                }
            )
        )
        .reset_index(drop=True)
    )

    # for each patient add delta features
    df_filled = (
        df_filled.groupby(patient_id_col)
        .apply(
            lambda x: x.assign(
                **{
                    f"{col}_delta_{i}": x[col].diff(i)
                    for i in range(output_length, 13)
                    for col in vital_signs
                }
            )
        )
        .reset_index(drop=True)
    )

    # fill missing values with 0
    df_filled = df_filled.fillna(0)

    return df_filled


def create_target(df, temporary_target_col, target_col, is_train):
    # keep data until index (i+6), where i is the first aoocurance of sepsis,
    # if all target values are 0, then keep all
    ind_first_sepsis = (
        df[df[temporary_target_col] == 1].index[0]
        if (df[temporary_target_col] == 1).any()
        else None
    )
    if ind_first_sepsis:
        df[target_col] = df[temporary_target_col].shift(-6)

        ind_first_sepsis = df[df[target_col] == 1].index[0]

        if not is_train:
            df = df.loc[: ind_first_sepsis + 6]
        df.loc[ind_first_sepsis:, target_col] = 1
    else:
        df[target_col] = df[temporary_target_col]
    return df.drop(columns=[temporary_target_col])


def prepare_data(df, patient_id_col, temporary_target_col, target_col, features, is_train):
    df = df[features + [patient_id_col, temporary_target_col]]
    df_shifted = (
        df.groupby(patient_id_col)
        .apply(
            lambda x: create_target(
                df=x,
                temporary_target_col=temporary_target_col,
                target_col=target_col,
                is_train=is_train,
            )
        )
        .reset_index(drop=True)
    )

    return df_shifted
