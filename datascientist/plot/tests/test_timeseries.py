from datetime import datetime


def valid_date(text):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d',
                '%Y.%m.%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('Provide a valid date time column name.')


def column_determine(df, x, y, date_column):
    flag = False
    if x is None:
        x = date_column
        flag = True
    if y is None:
        if flag is True:
            for i in df.columns:
                if df[i].dtypes == 'float64' or df[i].dtypes == 'float32':
                    y = i
                    break
            if y is None:
                for i in df.columns:
                    if df[i].dtypes == 'int64' or df[i].dtypes == 'int32':
                        y = i
                        break
            else:
                pass
            if y is None:
                raise ValueError("Should have atleast 1 column with either "
                                 "interger or float values.")
    else:
        y = date_column
    return x, y


def single_function(df):
    a = None
    for i in df.columns:
        if df[i].dtypes == 'float64' or df[i].dtypes == 'float32':
            a = i
            break
    if a is None:
        for i in df.columns:
            if df[i].dtypes == 'int64' or df[i].dtypes == 'int32':
                a = i
                break
    return a


def Array2D(df, date_column):
    df1 = df.loc[:, df.columns != date_column]
    return df1