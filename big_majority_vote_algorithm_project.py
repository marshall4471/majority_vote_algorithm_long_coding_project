#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.special import comb


# In[2]:


import math


# In[3]:


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) *
            error**k *
            (1-error)**(n_classifier - k)
            for k in range(k_start, n_classifier +1)]
    return sum(probs)


# In[ ]:





# In[4]:


ensemble_error(n_classifier=11, error=0.25)


# In[5]:


import numpy as np


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


error_range = np.arange(0.0, 1.01, 0.01)


# In[8]:


ens_errors = [ensemble_error(n_classifier=11, error=error)
             for error in error_range]


# In[9]:


plt.plot(error_range, ens_errors,
        label='Ensemble error',
        linewidth=2)
plt.plot(error_range, error_range,
        linestyle='--', label='Base error',
        linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()


# In[10]:


import numpy as np


# In[11]:


np.argmax(np.bincount([0, 0, 1],
                     weights=[0.2, 0.2, 0.6]))


# In[12]:


ex = np.array([[0.9, 0.1],
              [0.8, 0.2],
              [0.4, 0.6]])


# In[13]:


p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])


# In[14]:


p


# In[15]:


np.argmax(p)


# In[16]:


from sklearn.base import BaseEstimator


# In[17]:


from sklearn.base import ClassifierMixin


# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


from sklearn.base import clone


# In[20]:


from sklearn.pipeline import _name_estimators


# In[21]:


import numpy as np


# In[22]:


import operator


# In[23]:


class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    """A majority vote ensemble classifier
    
    Parameters
    classifier : array-like, shape = [n_classifiers]
     Different classifiers for the ensemble
     
     vote : str, {'classlabel', 'probability'}
       Default: 'classlabel'
       If'classlabel' the prediciton is based on
       the argmax of class labels. Else if
       'probability', the argmax of the sum of
       probabilites is used to predict the class label
       (recommended for calibrated classifiers).
       
    weights : array-like, shape = [n_classifiers]
      Optional, default: None
      If a list of 'int' or 'float' values are
      provided, the classifiers are weighted by
      importance; Uses uniform weight if 'weights=None'.
      
      """
    def __init__(self, classifiers,
                vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for
                                  key, value in
                                  _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    def fit(self, X, y):
        """Fit classifiers.
        
        Parameters
        X : {array-like, sparse matrix},
            shape = [n_examples, n_features]
            Matrix of training examples.
            
        y : array-like, shape = [n_examples]
            Vector of target class labels.
            
        Returns
        
        self : object
        
        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability'"
                            "or 'classlabel'; got (vote=%r)"
                            % self.vote)
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError("Number of classifier and weights"
                             "must be equal; got %d weights,"
                             "%d classifiers"
                             % (len(self.weights),
                             len(self.classifers)))
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
        
    def predict(self, X):
            """Predict class labels for X.
            
            Parameters
            X : {array-like, sparse matrix},
                Shape = [n_examples, n_features]
                Matrix of training examples.
                
            Returns
            
            maj_vote : array-like, shape = [n_examples]
                Predicted class labels.
                
                
            """
            if self.vote == 'probability':
                maj_vote = np.argmax(self.predict_proba(X), axis=1)
            else: # 'classlabel' vote
                
                predictions = np.asarray([clf.predict(X)
                                         for clf in
                                         self.classifiers_]).T
                maj_vote = np.apply_along_axis(lambda x: np.argmax(
                                               np.bincount(x,
                                               weights=self.weights)),
                                               axis=1,
                                               arr=predictions)
                maj_vote = self.lablenc_.inverse_transform(maj_vote)
            return maj_vote
            
    def predict_proba(self, X):
            """ Predict class probabilities for X.
            
            Parameters
            X : {array-like, sparse matrix},
                shape = [n_examples, n_features]
                Training vectors, where
                n_examples is the number of examples and
                n_features is the number of features.
                
            Returns
            
          avg_proba : array-like,
                shape = [n_examples, n_classes]
                Weighted average probability for
                each class per example.
                
            """
            probas = np.asarray([clf.predict_proba(X)
                            for clf in self.classifiers_])
            avg_proba = np.average(probas, axis=0,
                                  weights=self.weights)
            return avg_proba
        
    def get_params(self, deep=True):
            """Get classifier parameter names for GridSearch"""
            if not deep:
                return super(MajorityVoteClassifier,
                            self).get_params(deep=False)
            else:
                out = self.named_classifiers.copy()
                for name, step in self.named_classifiers.items():
                    for key, value in step.get_params(
                            deep=True).items():
                        out['%s__%s' % (name, key)] = value
            return out
            


# In[24]:


from sklearn import datasets


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


from sklearn.preprocessing import LabelEncoder


# In[28]:


iris = datasets.load_iris()


# In[29]:


X, y = iris.data[50:, [1, 2]], iris.target[50:]


# In[30]:


le = LabelEncoder()


# In[31]:


y = le.fit_transform(y)


# In[32]:


X_train, X_test, y_train, y_test =   train_test_split(X, y,
                   test_size=0.5,
                   random_state=1,
                   stratify=y)


# In[33]:


from sklearn.model_selection import cross_val_score


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


from sklearn.pipeline import Pipeline


# In[38]:


import numpy as np


# In[39]:


clf1 = LogisticRegression(penalty='l2',
                         C=0.01,
                         solver='lbfgs',
                         random_state=1)


# In[40]:


clf2 = DecisionTreeClassifier(max_depth=1,
                             criterion='entropy',
                             random_state=0)


# In[41]:


clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
                           


# In[42]:


pipe1 = Pipeline([['sc', StandardScaler()],
                ['clf', clf1]])


# In[43]:


pipe3 = Pipeline([['sc', StandardScaler()],
                 ['clf', clf3]])


# In[44]:


clf_labels = ['Logistic regression', 'Decision tree', 'KNN']


# In[ ]:





# In[45]:


print('10-fold cross validation:\n')


# In[46]:


for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                            X=X_train,
                            y=y_train,
                            cv=10,
                            scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.02f [%s]"
         % (scores.mean(), scores.std(), label))


# In[47]:


mv_clf = MajorityVoteClassifier(
                  classifiers=[pipe1, clf2, pipe3])


# In[48]:


clf_labels += ['Majority voting']


# In[49]:


all_clf = [pipe1, clf2, pipe3, mv_clf]


# In[50]:


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                            X=X_train,
                            y=y_train,
                            cv=10,
                            scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
           % (scores.mean(), scores.std(), label))


# In[51]:


from sklearn.metrics import roc_curve


# In[52]:


from sklearn.metrics import auc


# In[53]:


colors =['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls     in zip(all_clf, clf_labels, colors, linestyles):
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                    y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    
    plt.plot(fpr, tpr,
            color=clr,
            linestyle=ls,
            label='%s (auc = %0.2f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1], [0, 1],
            linestyle='--',
            color='gray',
            linewidth=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(alpha=0.5)
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.show()
    
    


# In[54]:


sc = StandardScaler()


# In[55]:


X_train_std = sc.fit_transform(X_train)


# In[56]:


from itertools import product


# In[57]:


x_min = X_train_std[:, 0].min() - 1


# In[58]:


x_max = X_train_std[:, 0].max() + 1


# In[59]:


y_min = X_train_std[:, 1].max() - 1


# In[60]:


y_max = X_train_std[:, 1].max() + 1


# In[61]:


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


# In[62]:


f, axarr = plt.subplots(nrows=2, ncols=2,
                      sharex='col',
                      sharey='row',
                      figsize=(7, 5))
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                       all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                   X_train_std[y_train==0, 1],
                                   c='blue',
                                   marker='^',
                                   s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                  X_train_std[y_train==0, 1],
                                  c='green',
                                  marker='o',
                                  s=50)
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(-3.5, -5.,
        s='Sepal width [standardized]',
        ha='center', va='center', fontsize=12) 
plt.text(-12.5, 4.5,
        s='Petal length [standardized]',
        ha='center', va='center',
        fontsize=12, rotation=90)
plt.show()


# In[63]:


mv_clf.get_params()


# In[64]:


import pandas as pd


# In[65]:


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases/wine/wine.data',
                      header=None)


# In[66]:


df_wine.columns = ['Class label', 'Alcohol',
                  'Malic acid', 'Ash',
                   'Alchalinity of Ash',
                  'Magnesium', 'Total phenols',
                  'Flavanoids', 'Nonflavanoid phenols',
                   'Prolanthocyanins',
                  'Color intensity', 'Hue',
                  'OD280/OD315 of diluted wines',
                  'Proline']


# In[67]:


# drop 1 class


# In[68]:


df_wine = df_wine[df_wine['Class label'] !=1]


# In[69]:


y = df_wine['Class label'].values


# In[70]:


X = df_wine[['Alcohol',
            'OD280/OD315 of diluted wines']].values


# In[71]:


from sklearn.preprocessing import LabelEncoder


# In[72]:


from sklearn.model_selection import train_test_split


# In[73]:


le = LabelEncoder()


# In[74]:


y = le.fit_transform(y)


# In[75]:


X_train, X_test, y_train, y_test =          train_test_split(X, y,
                          test_size=0.2,
                          random_state=1,
                          stratify=y)


# In[76]:


from sklearn.ensemble import BaggingClassifier


# In[77]:


tree = DecisionTreeClassifier(criterion='entropy',
                             random_state=1,
                             max_depth=None)


# In[78]:


bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)


# In[79]:


from sklearn.metrics import accuracy_score


# In[80]:


tree = tree.fit(X_train, y_train)


# In[81]:


y_train_pred = tree.predict(X_train)


# In[82]:


y_test_pred = tree.predict(X_test)


# In[83]:


tree_train = accuracy_score(y_train, y_train_pred)


# In[84]:


tree_test = accuracy_score(y_test, y_test_pred)


# In[85]:


print('Decision tree train/test accuracies %.3f/%.3f'
     % (tree_train, tree_test))


# In[ ]:





# In[86]:


bag = bag.fit(X_train, y_train)


# In[87]:


y_train_pred = bag.predict(X_train)


# In[88]:


y_test_pred = bag.predict(X_test)


# In[89]:


bag_train = accuracy_score(y_train, y_train_pred)


# In[90]:


bag_test = accuracy_score(y_test, y_test_pred)


# In[91]:


print('Bagging train/test accuracies %.3f/%.3f'
     % (bag_train, bag_test))


# In[92]:


x_min = X_train[:, 0].min() - 1


# In[93]:


x_max = X_train[:, 0].max() + 1


# In[94]:


y_min = X_train[:, 1].min() - 1


# In[95]:


y_max = X_train[:, 1].max() + 1


# In[96]:


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1],
                        [tree, bag],
                        ['Decision tree', 'Bagging']):
    clf.fit(X_train, y_train)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                       X_train[y_train==0, 1],
                       c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                       X_train[y_train==1, 1],
                       c='green', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.tight_layout()
plt.text(0, -0.2,
         s='OD280/OD315 of diluted wines',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()


# In[97]:


from sklearn.ensemble import AdaBoostClassifier


# In[98]:


tree = DecisionTreeClassifier(criterion='entropy',
                             random_state=1,
                             max_depth=1)


# In[99]:


ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=.1,
                         random_state=1)


# In[100]:


tree = tree.fit(X_train, y_train)


# In[101]:


y_train_pred = tree.predict(X_train)


# In[ ]:





# In[ ]:





# In[102]:


y_test_pred = tree.predict(X_test)


# In[103]:


tree_train = accuracy_score(y_train, y_train_pred)


# In[104]:


tree_test = accuracy_score(y_test, y_test_pred)


# In[105]:


print('Decision tree train/test accuracies %.3f/%.3f'
      %(tree_train, tree_test))


# In[106]:


ada = ada.fit(X_train, y_train)


# In[107]:


y_train_pred = ada.predict(X_train)


# In[108]:


y_test_pred = ada.predict(X_test)


# In[109]:


ada_train = accuracy_score(y_train, y_train_pred)


# In[110]:


ada_test = accuracy_score(y_test, y_test_pred)


# In[111]:


print('AdaBoost train/test accuracies %.3f/%.3f'
      % (ada_train, ada_test))


# In[112]:


x_min = X_train[:, 0].min() - 1


# In[113]:


x_max = X_train[:, 0].max() + 1


# In[114]:


y_min = X_train[:, 1].min() - 1


# In[115]:


y_max = X_train[:, 1].max() + 1


# In[116]:


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1],
                        [tree, ada],
                        ['Decision tree', 'AdaBoost']):
    clf.fit(X_train, y_train)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                       X_train[y_train==0, 1],
                       c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                       X_train[y_train==1, 1],
                       c='green', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.tight_layout()
plt.text(0, -0.2,
         s='OD280/OD315 of diluted wines',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()

