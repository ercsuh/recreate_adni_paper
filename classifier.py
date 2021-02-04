from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from plotnine import ggplot, aes, geom_bar, coord_flip
import pandas as pd
import numpy as np
import os


def classifiers(X, y):
    """
    Performs train/test split and tests multiple sklearn classifiers. 
    @param X: features
    @param y: class
    """
    (
        X_train, X_test, 
        y_train, y_test
    ) = train_test_split(
            X, y,
            test_size = 0.25,
            random_state = 0
        )

    classifiers = [
        LogisticRegression(),
        KNeighborsClassifier(),
        SVC(kernel='rbf', C=0.025),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        Lasso(),
        GaussianNB()
    ]

    labels = np.array(y.values.tolist())
    auc_df = pd.DataFrame(columns=['classifier', 'auc'])

    for clf in classifiers:
        name = clf.__class__.__name__

        clf.fit(X_train, y_train)
        predictions = clf.predict(X)
        fp, tp, _ = roc_curve(labels, predictions)
        roc_auc = auc(fp, tp)

        auc_df = auc_df.append(
            {
                'classifier': name,
                'auc': roc_auc
            }, 
            ignore_index=True
        )
    
    return auc_df


def plot_aucs(df):
    """
    Plots each classifier's AUC results.

    """
    plot = (
        ggplot(df)
        + geom_bar(aes(x='classifier', y='auc'), stat="identity")
        + coord_flip()
    )
    plot.save(filename='aucs.png')


if __name__ == "__main__":
    cwd = os.getcwd() 
    df = pd.read_csv(os.path.join(cwd, "total_data.tsv"), sep="\t")

    df.PTGENDER = pd.Categorical(df.PTGENDER)
    df.PTGENDER = df.PTGENDER.cat.codes
    X = df[['PTGENDER', 'PTEDUCAT', 'AGE']]
    y = df['Y']

    auc_df = classifiers(X, y)
    plot_aucs(auc_df)
