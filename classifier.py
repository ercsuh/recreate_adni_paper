from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os

if __name__ == "__main__":
    cwd = os.getcwd() 
    df = pd.read_csv(os.path.join(cwd, "total_data.tsv"), sep="\t")

    df.PTGENDER = pd.Categorical(df.PTGENDER)
    df.PTGENDER = df.PTGENDER.cat.codes
    # print(df.PTGENDER)
    # X = df.loc[:, ['PTGENDER', 'PTEDUCAT', 'AGE']]
    # y = df.loc[:, ['Y']]
    X = df[['PTGENDER', 'PTEDUCAT', 'AGE']]
    y = df['Y']
    (
        X_train, X_test, 
        y_train, y_test
    ) = train_test_split(
            X, y,
            test_size = 0.25,
            random_state = 0
        )

    print(y_test)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(y_test)

    score = lr.score(x_test, y_test)
    print(score)