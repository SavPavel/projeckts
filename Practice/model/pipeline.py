import datetime
import pandas as pd
import dill

from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    print('Homework predict')

    df = pd.read_csv('data/homework.csv')

    def filter_data(df):
        columns_to_drop = [
            'id',
            'url',
            'region',
            'region_url',
            'price',
            'manufacturer',
            'image_url',
            'description',
            'posting_date',
            'lat',
            'long'
        ]

        return df.drop(columns_to_drop, axis=1)

    def remove_outliers(df_remove):
        df_remove = df_remove.copy()

        def calculate_outliers(data):
            q25 = data.quantile(0.25)
            q75 = data.quantile(0.75)
            iqr = q75 - q25
            boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

            return boundaries

        boundaries_year = calculate_outliers(df_remove['year'])

        df_remove.loc[df_remove['year'] < boundaries_year[0], 'year'] = round(boundaries_year[0])
        df_remove.loc[df_remove['year'] > boundaries_year[1], 'year'] = round(boundaries_year[1])

        return df_remove

    def new_features(df_feat):
        df_feat = df_feat.copy()

        def short_model(x):
            if not pd.isna(x):
                return x.lower().split(' ')[0]
            else:
                return x

        df_feat.loc[:, 'short_model'] = df_feat['model'].apply(short_model)
        df_feat.loc[:, 'age_category'] = df_feat['year'].apply(
            lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

        return df_feat

    X = df.drop(columns=['price_category'], axis=1)
    y = df['price_category']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    por = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('remove_outliers', FunctionTransformer(remove_outliers)),
        ('boundaries_year', FunctionTransformer(new_features))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorial', categorical_transformer, make_column_selector(dtype_include=['object'])),
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('por', por),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best_model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score.mean():.4f}')
    with open('cars_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'loan prediction pipeline',
                'author': 'Savin Pavel',
                'version': 1,
                'data': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score}
            }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
