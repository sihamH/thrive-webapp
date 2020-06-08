import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import pickle
from keras import layers
import keras
from keras.models import Sequential

import joblib

import warnings
warnings.filterwarnings('ignore')
from pyprojroot import here

############################## variables ##################################
save_model = True
csvpath = str(here() / 'data/csv/dataset.csv')
modelpath = str(here() / 'models/')

################################ Data Prep ################################
# import csv
df_train = pd.read_csv(csvpath)
df_val = pd.read_csv(csvpath)

# separate traning set and validation set
for i in range(26,41):
    nameH = 'H'+str(i)
    nameP = 'P'+str(i)
    df_train = df_train[~df_train.filename.str.contains(nameH)]
    df_train = df_train[~df_train.filename.str.contains(nameP)]
       
df_val = pd.merge(df_val, df_train, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

# Dropping unneccesary columns
df_train = df_train.drop(['filename'],axis=1)
df_val = df_val.drop(['filename'],axis=1)

# pairewise Correlation 
corr_df = df_train.corr(method='pearson')
mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True

ax = plt.figure(figsize = (20,15))
sns.heatmap(corr_df, cmap = 'RdYlGn_r', mask=mask, linewidths=2.5) #annot = True,
plt.title('Pairewise correlation of all columns')


################################ Modeling ################################

def prep_for_modeling(df):

    #Encoding the Labels
    labels = df.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    #Scaling the Feature columns
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, :-1], dtype = float))

    return X, y

X_train, y_train = prep_for_modeling(df_train)
X_val, y_val = prep_for_modeling(df_val)


#Dividing data into training and Testing set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#---------------------------- Neural Net Model ----------------------------

def plot_acc(hist, kind):

    plt.figure()
    k_val = 'val_' + kind
    plt.plot(hist.history[kind])
    plt.plot(hist.history[k_val])
    plt.title('model ' + kind)
    plt.ylabel(kind)
    plt.xlabel('epoch')
    plt.ylim([0,1])
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

DLmodel = Sequential()
DLmodel.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
DLmodel.add(layers.Dense(32, activation='relu'))
DLmodel.add(layers.Dense(16, activation='relu'))
DLmodel.add(layers.Dense(1, activation='sigmoid'))
DLmodel.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = DLmodel.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=128, validation_data = (X_val, y_val))    

#print(DLmodel.predict(X_train[0:1]))

# Plot accuracy and loss
plot_acc(history, 'acc')
plot_acc(history, 'loss')

#---------------------------- Logistic Regression Model ----------------------------

# Model
logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(X_train, y_train)
print('Logistic Regression score {}'.format(logreg.score(X_val,y_val)))

# Prediction
y_pred = logreg.predict(X_val)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_val, y_val)))

# confusion matrix
cm = confusion_matrix(y_val, y_pred)
print(cm)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (6,5))
sns.heatmap(cm_normalized, cmap = 'gray', linewidths=2.5, annot=True) #annot = True,
plt.title('Confusion Matrix')
plt.show()

# Precision and recall
print(classification_report(y_val, y_pred))

# ROC curve
logit_roc_auc = roc_auc_score(y_val, logreg.predict(X_val))
fpr, tpr, thresholds = roc_curve(y_val, logreg.predict_proba(X_val)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
#plt.show()

################################ Save models ################################

# DL
if save_model:
    filenameDL = modelpath+'/DLmodel.h5'
    print(filenameDL)
    DLmodel.save(filenameDL)


# LR
if save_model:
    filenameLR = modelpath+'/LR_model.sav'
    pickle.dump(logreg, open(filenameLR, 'wb'))