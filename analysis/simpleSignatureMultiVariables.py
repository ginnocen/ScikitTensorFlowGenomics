import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets, neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

doplot1D=0
doplot2D=0
doMLMultivariables=1
doML2variables=0

h = .02  # step size in the mesh

datasets= ["small sample"]

names = ["GradientBoostingClassifier","Nearest_Neighbors", "Linear_SVM", "RBF_SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA"]

classifiers = [
        GradientBoostingClassifier(learning_rate=0.01, n_estimators=2500, max_depth=1),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, probability=True),
        SVC(gamma=2, C=1,probability=True),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]


input_file = "../samples/msk_simulations/msk_train.csv"
input_file_test = "../samples/msk_simulations/msk_test.csv"
df = pd.read_csv(input_file, header = 0)
df_test= pd.read_csv(input_file_test, header = 0)

bin_values = np.arange(start=-0.2, stop=1.2, step=0.03)

if(doplot1D == 1): 
    df.Signature_3_ml.hist(bins=bin_values,color="red",label="Sign3_ml")
    df.Signature_3_c.hist(bins=bin_values, color="green", label="Sign3_c")
    plt.title("Signature 3_ml and 3_c")
    plt.xlabel("Value")
    plt.ylabel("Entries")
    plt.legend()
    plt.show()
    
if(doplot2D == 1): 
    df_sig=df.loc[df['signal'] ==1]                              
    df_bkg=df.loc[df['signal'] ==0]                              
    plt.scatter(df_sig.Signature_3_c, df_sig.Signature_3_ml, s=20, c='b', label='signal')
    plt.scatter(df_bkg.Signature_3_c, df_bkg.Signature_3_ml, s=20, c='r', label='bkg')
    plt.title("Signature 3_ml vs 3_c")
    plt.xlabel("Signature_3_c")
    plt.ylabel("Signature_3_ml")
    plt.legend()
    plt.show()
 
 
if(doMLMultivariables == 1): 

    y = np.asarray(df.signal)
    df_selected = df.drop(['tumor', 'signal', "Signature_3_ml","exp_sig3","total_snvs"], axis=1)
    df_features = df_selected.to_dict(orient='records')
    vec = DictVectorizer()
    X = vec.fit_transform(df_features).toarray()

    y_test = np.asarray(df_test.signal)
    df_selected_test = df_test.drop(['tumor', 'signal', "Signature_3_ml","exp_sig3","total_snvs"], axis=1)
    df_features_test = df_selected_test.to_dict(orient='records')
    vec_test = DictVectorizer()
    X_test = vec_test.fit_transform(df_features_test).toarray()

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    X_train=X
    y_train=y

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
                         
    i=1
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train,)
        score = clf.score(X_test, y_test)
        print score
        y_test_prediction=clf.predict(X_test)
        y_test_probability=clf.predict_proba(X_test)[:,1]
    
        df_test_pred=df_test
        df_test_pred = df_test_pred.assign(estimated_decision=pd.Series(y_test_prediction))
        df_test_pred = df_test_pred.assign(estimated_probability=pd.Series(y_test_probability))
        nameoutput="predictions/cvspredictionsMultiVar"+names[i-1]+".csv"
        df_test_pred.to_csv(nameoutput)
        i += 1
    
if(doML2variables == 1): 
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    y = np.asarray(df.signal)
    df_selected = df.drop(['tumor', 'signal', "Signature_3_l_rat", "Signature_3_wl","rat_sig3","Signature_other_c4_ml","exp_sig3","total_snvs","Signature_3_c1_ml","Signature_3_c2_ml","Signature_3_c3_ml","Signature_3_c4_ml"], axis=1)
    df_features = df_selected.to_dict(orient='records')
    vec = DictVectorizer()
    X = vec.fit_transform(df_features).toarray()

    y_test = np.asarray(df_test.signal)
    df_selected_test = df_test.drop(['tumor', 'signal', "Signature_3_l_rat", "Signature_3_wl","rat_sig3","Signature_other_c4_ml","exp_sig3","total_snvs","Signature_3_c1_ml","Signature_3_c2_ml","Signature_3_c3_ml","Signature_3_c4_ml"], axis=1)
    df_features_test = df_selected_test.to_dict(orient='records')
    vec_test = DictVectorizer()
    X_test = vec_test.fit_transform(df_features_test).toarray()

    X_train=X
    y_train=y

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
                         
    figure = plt.figure(figsize=(10,3))
    i=1
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(1, len(classifiers) + 1, i)
        clf.fit(X_train, y_train,)
        score = clf.score(X_test, y_test)
        y_test_prediction=clf.predict(X_test)
        y_test_probability=clf.predict_proba(X_test)[:,1]
    
        df_test_pred=df_test
        df_test_pred = df_test_pred.assign(estimated_decision=pd.Series(y_test_prediction))
        df_test_pred = df_test_pred.assign(estimated_probability=pd.Series(y_test_probability))
        nameoutput="predictions/cvspredictions"+names[i-1]+".csv"
        df_test_pred.to_csv(nameoutput)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
        i += 1
    
plt.show()


