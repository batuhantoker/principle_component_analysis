import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pandas as pd

def unique(list1):
  # initialize a null list
  unique_list = []

  # traverse for all elements
  for x in list1:
    # check if exists in unique_list or not
    if x not in unique_list:
      unique_list.append(x)
  # print list
  for x in unique_list:
    print(x)
directories=glob.glob("data/Exercise_3/*.mat")
print(directories)
exer='Exercise C'
mat = scipy.io.loadmat(directories[2])
mat_ks = [k for k in mat.keys()]

merged_emg = np.empty((0,16), int)
merged_glove = np.empty((0,22), int)
merged_restimulus = np.empty((0, 1), int)
for i in directories:
  to_merge=scipy.io.loadmat(i)
  merged_emg = np.append(merged_emg, to_merge['emg'],axis = 0)
  merged_glove = np.append(merged_glove, to_merge['glove'], axis=0)
  merged_restimulus = np.append(merged_restimulus, to_merge['restimulus'])
emg1,emg2=np.hsplit(merged_emg,2)
merged_emg2=np.concatenate((emg1, emg2))
merged_restimulus2=np.concatenate((merged_restimulus, merged_restimulus))
print(unique(merged_restimulus))

## PCA analysis
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns

x= merged_glove #[::3,::3]
y= merged_restimulus #[::3]
x = pd.DataFrame(data=x, columns=[str(y) for y in range(1,23)]) #For glove
#x = pd.DataFrame(data=x, columns=[str(y) for y in range(1,17)]) #For emg
#x = pd.DataFrame(data=x, columns=[str(y) for y in range(1,9)]) #For emg2

pca_x = PCA()
principalComponents_x = pca_x.fit_transform(x)
## Explained variance

exp_var_pca = pca_x.explained_variance_ratio_
#
# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
#
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
print(cum_sum_eigenvalues)
#
# Create the visualization plot
#
plt.figure(0)
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.title(exer)
# Save the figure as '.eps' file.
plt.savefig('pca_variance.png', format='png', dpi=1200,bbox_inches='tight')

## For 2 components
pca_x = PCA(n_components=2)

principalComponents_x = pca_x.fit_transform(x)
print(abs( pca_x.components_ ))

principal_x_Df = pd.DataFrame(data = principalComponents_x , columns = ['principal component 1', 'principal component 2'])
principal_x_Df.tail()
pca_data_vis = np.vstack((principalComponents_x.T,y)).T
pca_vis_df = pd.DataFrame(data=pca_data_vis,columns=("1st principal","2nd principal","label"))

sns.FacetGrid(pca_vis_df,hue="label",size=6).map(plt.scatter,'1st principal','2nd principal', s=0.01).add_legend()
plt.title(exer)
plt.savefig('pca.png', format='png', dpi=1200,bbox_inches='tight')

# number of components
n_pcs= pca_x.components_.shape[0]
# get the index of the most important feature on EACH component i.e. largest absolute value
# using LIST COMPREHENSION HERE
most_important = [np.abs(pca_x.components_[i]).argmax() for i in range(n_pcs)]


initial_feature_names = x.columns

# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# using LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.DataFrame(sorted(dic.items()))
print(df)

# pca2 = PCA(n_components=3)
# pca_result = pca2.fit_transform(x)
#
# plt.figure(2)
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=pca_result[:,0],
#     ys=pca_result[:,1],
#     zs=pca_result[:,2],
#     c=y,
#     cmap='tab10'
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
plt.show()


#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
#print(x_train.shape)


