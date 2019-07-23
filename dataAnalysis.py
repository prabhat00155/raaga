from audioFeatureExtraction import loadDataset
import numpy as np



def getDataset():
    X = []
    Y = []
    for row in loadDataset():
        for target, feature in row.items():
            X.append(feature)
            Y.append(int(target == 'mohanam'))
    return X, Y

def getTrainTestSplit(X, Y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, Y, test_size=0.33, random_state=42)

def doPCA(X, Y):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from pprint import pprint

    pca = PCA()
    X_r = pca.fit_transform(X)
    

def buildSGDPipeline(X,y):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    logistic = SGDClassifier(max_iter=10000, tol=1e-5, random_state=0)
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    param_grid = {
    'pca__n_components': [2, 6, 12, 24, 36],
    'logistic__alpha': np.logspace(-6,-4, 4, 5),
    'logistic__loss':['hinge', 'log', 'modified_huber', 'perceptron'],
    'logistic__penalty':['l2', 'l1', 'elasticnet']
    }
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5)
    search.fit(X, y)
    return search

def getGradientBoosting(X,y):
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    pipe = Pipeline(steps = [('GBC', clf)])
    param_grid = {
        'GBC__loss':["deviance", "exponential"],
        'GBC__learning_rate':[0.1, 0.001, 1, 0.0001],
        'GBC__n_estimators':[100, 1000, 500, 50],
        'GBC__max_depth':[3,6,12],
        'GBC__max_features':["auto", "sqrt", "log2", None],
    }
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5)
    search.fit(X,y)
    return search

if __name__ == "__main__":
    X, Y = getDataset()
    X_train, X_test, y_train, y_test = getTrainTestSplit(X, Y)
    gridSearch = getGradientBoosting(X_train,y_train)
    score = gridSearch.score
    bestEstimator = gridSearch.best_estimator_
    bestParameters = gridSearch.best_params_