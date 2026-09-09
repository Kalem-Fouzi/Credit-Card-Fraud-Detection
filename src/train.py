from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config as c
from src import data_loader as dl

X_train, X_test, y_train, y_test = dl.load_and_split()

baselines = {
    'LogReg': LogisticRegression(max_iter=1000, random_state=c.RANDOM_STATE),
    'DT': DecisionTreeClassifier(random_state=c.RANDOM_STATE),
    'LightGBM': LGBMClassifier(random_state=c.RANDOM_STATE)
}
def evaluate(type, model, X_train, X_test):

    scale = StandardScaler() if type == 'LogReg' else 'passthrough'
    pipeline = Pipeline([
        ('sclaer', scale),
        ('classifier', model)
    ]
    )
    pipeline.fit(X_train, y_train)
    test_preds = pipeline.predict(X_test)
    train_preds = pipeline.predict(X_train)
    print('test score', f1_score(y_test, test_preds))
    print('train score', f1_score(y_train, train_preds))
    print(classification_report(y_test, test_preds))

for name, model in baselines.items():
    print(f'{name} model')
    evaluate(name, model, X_train, X_test)

def cvs_pipeline(type, model, X, y, cv=c.CV, method='f1'):

    scale = StandardScaler() if type == 'LogReg' else 'passthrough'
    pipe = Pipeline([
        ('sclaer', scale),
        ('classifier', model)
    ]
    )
    val = cross_val_score(pipe, X, y, cv=cv, scoring=method)
    print('CVS SCORES:')
    print('mean ', val.mean())
    print('std ', val.std())

for name, model in baselines.items():
    print(f'{name} model')
    cvs_pipeline(name, model, X_train, y_train) 
    