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

do1D=0
do2D=0
doSVM=1

h = .02  # step size in the mesh

datasets= ["small sample"]

names = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA"]


input_file = "msk_tune_for_gian2.csv"
input_file_test = "msk_test.csv"
df = pd.read_csv(input_file, header = 0)
df_test= pd.read_csv(input_file_test, header = 0)

bin_values = np.arange(start=-0.2, stop=1.2, step=0.03)

if(do1D == 1): 
    df.Signature_3_ml.hist(bins=bin_values,color="red",label="Sign3_ml")
    df.Signature_3_c.hist(bins=bin_values, color="green", label="Sign3_c")
    plt.title("Signature 3_ml and 3_c")
    plt.xlabel("Value")
    plt.ylabel("Entries")
    plt.legend()
    plt.show()
    
if(do2D == 1): 
    df_sig=df.loc[df['signal'] ==1]                              
    df_bkg=df.loc[df['signal'] ==0]                              
    plt.scatter(df_sig.Signature_3_c, df_sig.Signature_3_ml, s=20, c='b', label='signal')
    plt.scatter(df_bkg.Signature_3_c, df_bkg.Signature_3_ml, s=20, c='r', label='bkg')
    plt.title("Signature 3_ml vs 3_c")
    plt.xlabel("Signature_3_c")
    plt.ylabel("Signature_3_ml")
    plt.legend()
    plt.show()
 
y = np.asarray(df.signal)
df_selected = df.drop(['tumor', 'signal', "Signature_3_ml","exp_sig3","total_snvs","Signature_clock_c1_ml"], axis=1)
df_features = df_selected.to_dict(orient='records')
vec = DictVectorizer()
X = vec.fit_transform(df_features).toarray()

y_test = np.asarray(df_test.signal)
df_selected_test = df_test.drop(['tumor', 'signal', "Signature_3_ml","exp_sig3","total_snvs"], axis=1)
df_features_test = df_selected_test.to_dict(orient='records')
vec_test = DictVectorizer()
X_test = vec.fit_transform(df_features_test).toarray()

classifiers = [
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

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
X_train=X
y_train=y

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
                         


cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
ax1.set_title('Training sample')
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel("Signature_3_c")
ax1.set_ylabel("Signature_3_ml")

ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
ax2.set_title('Testing sample')
ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel("Signature_3_c")
ax2.set_ylabel("Signature_3_ml")


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
    
plt.show()




