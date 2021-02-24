from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from plotnine import ggplot, aes, geom_bar, coord_flip
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def classifiers(X, y):
    """
    Performs train/test split and tests multiple sklearn classifiers. 
    @param X: features
    @param y: class
    """
    # (
    #     X_train, X_test, 
    #     y_train, y_test
    # ) = train_test_split(
    #         X, y,
    #         test_size = 0.25,
    #         random_state = 0
    #     )

    classifiers = [
        DummyClassifier(strategy="uniform", random_state=0),
        LogisticRegression(),
        KNeighborsClassifier(),
        SVC(kernel='rbf', C=0.025),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        # Lasso()
    ]

    X = np.array(X.values.tolist())
    y = np.array(y.values.tolist())
    auc_df = pd.DataFrame(columns=['classifier', 'auc'])

    for clf in classifiers:
        skf = StratifiedKFold(n_splits=5)
        name = clf.__class__.__name__
        auc_list = []
        #---
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        #---
        if name == "LogisticRegression":
            scaler = MinMaxScaler()
            scaler.fit(X)
            X1 = scaler.transform(X)
        else:
            X1 = X
        
        i = 1
        for train_index, test_index in skf.split(X1, y):
            X_train, X_test = X1[train_index], X1[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            # if name == "LogisticRegression":
            #     print(clf.coef_)
            #     print(clf.intercept_)
            predictions = clf.predict(X_test)
            fp, tp, _ = roc_curve(y_test, predictions)
            auc_list.append(auc(fp, tp))
            #---
            viz = plot_roc_curve(
                clf, X_test, y_test,
                name=f'ROC fold {i}',
                alpha=0.3, lw=1, ax=ax
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            i = i+1
        

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title=f"{name} ROC with CV")
        ax.legend(loc="lower right")
        plt.savefig(f"figures/{name}_roc.jpg")
        #---
        auc_df = auc_df.append(
            {
                'classifier': name,
                'auc': sum(auc_list) / len(auc_list)
            }, 
            ignore_index=True
        )
    print(auc_df)
    
    return auc_df


def plot_auc(df):
    """
    Plots each classifier's AUC results.
    """
    plot = (
        ggplot(df)
        + geom_bar(aes(x='classifier', y='auc'), stat="identity")
        + coord_flip()
    )
    plot.save(filename='figures/auc.png')


if __name__ == "__main__":
    cwd = os.getcwd() 
    df = pd.read_csv(os.path.join(cwd, "total_data.tsv"), sep="\t")

    df.PTGENDER = pd.Categorical(df.PTGENDER)
    df.PTGENDER = df.PTGENDER.cat.codes
    X = df[['PTGENDER', 'PTEDUCAT', 'AGE']]
    y = df['Y']

    auc_df = classifiers(X, y)
    plot_auc(auc_df)
