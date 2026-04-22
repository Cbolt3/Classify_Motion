import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)
from sklearn.metrics import (roc_curve, roc_auc_score, RocCurveDisplay)
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay


#STEP 3 VISUALIZATION:
    #visualize a few samples from dataset (all three axes and walking, jumping modes)
    #accel v time graph
    #additional modes of visualization
    #meta data visualization
names = ['Charlie', 'Maddy', 'Josh']
labels = ['x acceleration', 'y acceleration', 'z acceleration', 'abs acceleration']
colors = ['red', 'blue', 'green']
fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(20, 10)) #change rows back to 6 and cols back to 4

jump = pd.read_csv("jumping_data.csv")#raw jumping data
jump_x_values = np.arange(0, (len(jump)/100), 0.01) #fixing x axis
jump.iloc[:, 0] = jump_x_values

walk = pd.read_csv("walking_data.csv")#raw walking data
walk_x_values = np.arange(0, (len(walk)/100), 0.01) #fixing x axis
walk.iloc[:, 0] = walk_x_values

#.format(names[0])

#ax[0, 0].plot(jump.iloc[:, 0], jump.iloc[:, 4])
#ax[0, 0].set_title(labels[3])

"""for n, name in enumerate(names):
    for i in range(4):
        jump = pd.read_csv("{}_jumping.csv".format(name))
        if name == 'Josh':
            jump.iloc[:,0] = np.arrange(0, len(jump +1)
        ax[n*2, i].plot(jump.iloc[:, 0], jump.iloc[:, i+1], color= colors[n])
        ax[n*2, i].set_title(name + ' jump ' + labels[i])

        walk = pd.read_csv("{}_walking.csv".format(name))
        ax[n*2+1, i].plot(walk.iloc[:, 0], walk.iloc[:, i + 1], color= colors[n])
        ax[n*2+1, i].set_title(name + ' walk ' + labels[i])"""

#fig.tight_layout()
#plt.show()


#STEP 4 PRE PROCESSING
    #REMOVE NOISE WITH MOVING AVERAGE FILTER

ax[1].plot(walk.iloc[:, 0], walk.iloc[:, 1:])

jump_sma = jump.rolling(51).mean()
walk_sma = walk.rolling(51).mean()

ax[0].plot(walk_sma.iloc[:, 0], walk_sma.iloc[:, 1:])
ax[0].set_title("Rolling Average 601")
ax[0].set_ylabel('Acceleration (m/s^2)')
ax[0].set_xlabel('Time (s)')

#STEP 5 FEATURE EXTRACTION AND NORMALIZATION
    #from each 5 second time window extract a minimum of 10 features
        #jump data

jfeatures = pd.DataFrame(columns=['xmean', 'ymean', 'zmean', 'absmax', 'absmin', 'xvar', 'yvar', 'zvar', 'abskew', 'std'])
jfeatures['xmean'] = jump_sma.rolling(window=500).mean().iloc[:, 1]
jfeatures['ymean'] = jump_sma.rolling(window=500).mean().iloc[:, 2]
jfeatures['zmean'] = jump_sma.rolling(window=500).mean().iloc[:, 3]
jfeatures['absmax'] = jump_sma.rolling(window=500).max().iloc[:, 4]
jfeatures['absmin'] = jump_sma.rolling(window=500).min().iloc[:, 4]
jfeatures['xvar'] = jump_sma.rolling(window=500).var().iloc[:, 1]
jfeatures['yvar'] = jump_sma.rolling(window=500).var().iloc[:, 2]
jfeatures['zvar'] = jump_sma.rolling(window=500).var().iloc[:, 3]
jfeatures['abskew'] = jump_sma.rolling(window=500).skew().iloc[:, 4]
jfeatures['std'] = jump_sma.rolling(window=500).std().iloc[:, 4]

wfeatures = pd.DataFrame(columns=['xmean', 'ymean', 'zmean', 'absmax', 'absmin', 'xvar', 'yvar', 'zvar', 'abskew', 'std'])
wfeatures['xmean'] = walk_sma.rolling(window=500).mean().iloc[:, 1]
wfeatures['ymean'] = walk_sma.rolling(window=500).mean().iloc[:, 2]
wfeatures['zmean'] = walk_sma.rolling(window=500).mean().iloc[:, 3]
wfeatures['absmax'] = walk_sma.rolling(window=500).max().iloc[:, 4]
wfeatures['absmin'] = walk_sma.rolling(window=500).min().iloc[:, 4]
wfeatures['xvar'] = walk_sma.rolling(window=500).var().iloc[:, 1]
wfeatures['yvar'] = walk_sma.rolling(window=500).var().iloc[:, 2]
wfeatures['zvar'] = walk_sma.rolling(window=500).var().iloc[:, 3]
wfeatures['abskew'] = walk_sma.rolling(window=500).skew().iloc[:, 4]
wfeatures['std'] = walk_sma.rolling(window=500).std().iloc[:, 4]


#feature normalization
    #jump data
sc = StandardScaler()
jfeatures_scaled = sc.fit_transform(jfeatures)
jfeatures_scaled = pd.DataFrame(jfeatures_scaled, columns=jfeatures.columns)

    #walk data
sc = StandardScaler()
wfeatures_scaled = sc.fit_transform(wfeatures)
wfeatures_scaled = pd.DataFrame(wfeatures_scaled, columns=wfeatures.columns)

# Identify outliers as datapoints with a z-score greater than 3 and remove them
jdf = jump_sma.iloc[602:, :]
wdf = walk_sma.iloc[602:, :]

jz = np.abs(stats.zscore(jdf.iloc[:, 4]))#z score calculation
wz = np.abs(stats.zscore(wdf.iloc[:, 4]))

threshold = 3
jump_smooth = jdf[jz < threshold]
walk_smooth = wdf[wz < threshold]

#data normalization
jump_scaled = sc.fit_transform(jump_smooth)
walk_scaled = sc.fit_transform(walk_smooth)

#jfeatures["acceleration"] = jump_scaled[:, 4]
#wfeatures["acceleration"] = walk_scaled[:, 4]

#plt.show()

#STEP 6 CREATING CLASSIFIER
    #train logistic regression model to classify into classes of walking and jumping
#jumpdf = pd.DataFrame(jump_scaled)
#jumpdf.iloc[:, 5] = 1

jumpdf = jfeatures_scaled.iloc[1102:, :]
jumpdf["label"] = 1
print(jumpdf)

#jump = 1, walk = 0
#walkdf = pd.DataFrame(walk_scaled[:len(jumpdf), :])
#walkdf.iloc[:, 5] = 0

walkdf = wfeatures_scaled.iloc[1102:len(jumpdf)+1101, :]
walkdf["label"] = 0

print(len(jumpdf))
print(len(walkdf))

#jumpdf = jumpdf.drop(jumpdf.columns[0], axis=1)

#walkdf = walkdf.drop(walkdf.columns[0], axis=1)
ax[2].plot(walkdf)

result = pd.concat([jumpdf, walkdf])
print(result)

non_numeric_columns = result.select_dtypes(exclude=['number']).columns

# Loop through each non-numeric column and print its non-numeric entries
for column in non_numeric_columns:
    non_numeric_entries = result[result[column].apply(lambda x: not pd.api.types.is_numeric_dtype(x))]
    print(f"Non-numeric entries in column '{column}':")
    print(non_numeric_entries[column])


data = result.iloc[:, :-1]
nlabels = result.iloc[:, -1]
 
X_train, X_test, y_train, y_test = train_test_split(data, nlabels, test_size=0.1, random_state=0, shuffle=True)
scaler = StandardScaler()

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

# Train model
clf.fit(X_train, y_train)

# Obtain prediction and probability
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
print('\nConfusion Matrix:')
print(cm)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print('\nF1 Score: ', f1)

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# Calculating Auc
roc_auc = roc_auc_score(y_test, y_clf_prob[:, 1])
print('\nAUC: ', roc_auc)


#part2
pca = PCA(n_components=2)

# Create pipeline with StandardScaler() and pca
pca_pipe = make_pipeline(StandardScaler(), pca)

# Apply the pipeline to X_train and X_test
X_train_pca = pca_pipe.fit_transform(X_train)
X_test_pca = pca_pipe.transform(X_test)

# Create pipeline with only LogisticRegression
clf = make_pipeline(LogisticRegression(max_iter=10000))

# Train clf
clf.fit(X_train_pca, y_train)

# Obtain predictions for X_test_pca
y_pred_pca = clf.predict(X_test_pca)

# Calculate accuracy
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print('\nAccuracy with PCA: ',accuracy_pca)

# Create the decision boundary display
disp = DecisionBoundaryDisplay.from_estimator(clf, X_train_pca, response_method="predict", xlabel='X1', ylabel='X2',alpha=0.5)

# Scatter plot
disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.show()