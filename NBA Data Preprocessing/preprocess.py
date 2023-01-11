import os
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_data(data):
    df = pd.read_csv(data)

    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')

    df['team'].fillna("No Team", inplace=True)

    df["height"] = df["height"].str.extract(r"/ ([\d\.]+)").astype(float)
    df["weight"] = df["weight"].str.extract(r"/ ([\d\.]+)").astype(float)
    df["salary"] = df["salary"].str.replace("$", "", regex=False).astype(float)

    df["country"] = df["country"].apply(lambda x: "USA" if x == "USA" else "Not-USA")
    df.loc[df['draft_round'] == "Undrafted", 'draft_round'] = "0"
    return df


def feature_data(df):
    df['version'] = pd.to_datetime(df['version'].apply(lambda x: x.replace('NBA2k', '20')), format='%Y')
    df['age'] = pd.DatetimeIndex(df['version']).year - pd.DatetimeIndex(df['b_day']).year
    df['experience'] = pd.DatetimeIndex(df['version']).year - pd.DatetimeIndex(df['draft_year']).year
    df['bmi'] = df['weight'] / df['height'] ** 2
    df.drop(['version', 'b_day', 'draft_year', 'weight', 'height'], axis=1, inplace=True)
    for i in df.columns:
        if df[i].nunique() > 50 and i not in ['age', 'experience', 'bmi', 'salary']:
            df.drop(i, axis=1, inplace=True)
    return df


def multicol_data(df, target='salary'):
    corr_matrix = df.select_dtypes('number').drop(columns=target).corr()
    multicol_pairs = []
    for idx in corr_matrix.index:
        for column in corr_matrix.columns:
            if abs(corr_matrix[idx][column] > 0.7) and idx != column:
                ind_corr = corr_matrix[column][idx].round(6)
                multicol_pairs.append([idx, ind_corr])

    df.drop(columns=multicol_pairs[0][0], inplace=True)

    return df


def transform_data(df, target='salary'):
    num_feat_df = df.select_dtypes('number').drop(columns=target)
    cat_feat_df = df.select_dtypes('object')

    scaler = StandardScaler()
    encoder = OneHotEncoder().fit(cat_feat_df)

    col_names = encoder.categories_
    col_names_flat = [x for sublist in col_names for x in sublist]
    scaled_numerical = pd.DataFrame(scaler.fit_transform(num_feat_df), columns=num_feat_df.columns)

    encoded_nominal = encoder.transform(cat_feat_df)
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_nominal, columns=col_names_flat)

    x = pd.concat([scaled_numerical, encoded_df], axis=1)
    y = df[target]

    return x, y


def main():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    if 'nba2k-full.csv' not in os.listdir('../Data'):
        print('NBA dataset loading.')
        url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/nba2k-full.csv', 'wb').write(r.content)
        print('Loaded.')

    path = "../Data/nba2k-full.csv"

    df_cleaned = clean_data(path)
    df_featured = feature_data(df_cleaned)
    df = multicol_data(df_featured)
    x, y = transform_data(df)

    answer = {
        'shape': [x.shape, y.shape],
        'features': list(x.columns),
    }
    print(answer)


if __name__ == '__main__':
    main()
