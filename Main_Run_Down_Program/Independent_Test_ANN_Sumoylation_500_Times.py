import pandas as pd
import numpy as np

df = pd.read_csv("Thirty_Percent_CD_HIT_SUMOYLATION_TRAINING_SET.csv")

df = df.drop(["Unnamed: 0"],axis=1)

label = np.array(df["label"])

df_test = pd.read_csv("Thirty_Percent_CD_HIT_SUMOYLATION_Independent_Test_Data_SET.csv")

df_test = df_test.drop(["Unnamed: 0"],axis=1)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import *
from sklearn.metrics import roc_curve, roc_auc_score, classification_report,auc
import tensorflow.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Bidirectional,Dense, LSTM, Activation, Dropout, Flatten, LeakyReLU
from sklearn.metrics import accuracy_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM

from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Input, add, Conv1D, MaxPooling1D

from tensorflow.keras.layers import LSTM, GRU, SimpleRNN


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime

import os.path
from scipy.spatial import distance
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import KFold
import numpy as np
import random
from Bio import SeqIO
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras import backend as K
from tensorflow.keras.backend import expand_dims
import tensorflow as tf
import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras import backend as K
from tensorflow.keras.backend import expand_dims
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import *
from sklearn.metrics import roc_curve, roc_auc_score, classification_report,auc
import tensorflow.keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Bidirectional,Dense, LSTM, Activation, Dropout, Flatten, LeakyReLU
from sklearn.metrics import accuracy_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM

from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Input, add, Conv1D, MaxPooling1D

from tensorflow.keras.layers import LSTM, GRU, SimpleRNN


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime

import os.path
from scipy.spatial import distance
import scipy.io as sio

import os 

from sklearn.linear_model import LogisticRegression

import random 

import numpy as np 
import pandas as pd 
import os

for Andrew in range(500):


    df["label"].value_counts()

    train = df.iloc[:,3:-1]

    X_train = np.array(train)

    y_train = label

    print(X_train.shape,y_train.shape)

    seed = random.randint(1,10000000)

    print("***************************")
    print()
    print("Seed is   :",seed)
    print()
    print("***************************")

    print(X_train.shape,y_train.shape)

    rus = RandomUnderSampler(random_state=seed)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    print(X_train.shape)

    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    x_train, x_val, y_train_1, y_val = train_test_split(X_train, y_train,random_state =seed, test_size=0.1)

    y_train_1 = tf.keras.utils.to_categorical(y_train_1,2)
    y_val = tf.keras.utils.to_categorical(y_val,2)


    model = Sequential()
    model.add(Dense(64, input_dim=1024, kernel_initializer='uniform', activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))


    model.compile(optimizer=tf.keras.optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="ROC_ROC_Premise_Assumption.h5", 
                                    monitor = 'val_accuracy',
                                    verbose=0, 
                                    save_weights_only=False,
                                    save_best_only=True)

    reduce_lr_acc = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.001, patience=7, verbose=1, min_delta=1e-4, mode='max')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,mode='max')

    history = model.fit(x_train, y_train_1,epochs=400,verbose=1,batch_size=256,
                            callbacks=[checkpointer,reduce_lr_acc, early_stopping],validation_data=(x_val, y_val))


    test = df_test.iloc[:,3:-1]

    X_independent = np.array(test)

    y_independent = np.array(df_test["Label"])

    print(X_independent.shape,y_independent.shape)

    rus = RandomUnderSampler(random_state=seed)
    X_independent, y_independent = rus.fit_resample(X_independent, y_independent)
    print(X_independent.shape,y_independent.shape)

    print(X_independent.shape,y_independent.shape)

    yval = y_independent

    Y_pred = model.predict(X_independent)
    Y_pred = (Y_pred > 0.5)
    y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]
    y_pred = np.array(y_pred)

    confusion = confusion_matrix(yval,y_pred)

    print("Matthews Correlation : ",matthews_corrcoef(y_independent, y_pred))
    print("Confusion Matrix : \n",confusion_matrix(y_independent, y_pred))
    print("Accuracy on test set:   ",accuracy_score(y_independent, y_pred))

    cm = confusion_matrix(y_independent, y_pred)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    mcc = matthews_corrcoef(y_independent, y_pred)

    Sensitivity = TP/(TP+FN)

    Specificity = TN/(TN+FP)

    print("Sensitivity:   ",Sensitivity,"\t","Specificity:   ",Specificity)

    print(classification_report(y_independent, y_pred))

    fpr, tpr, _ = roc_curve(y_independent, y_pred)

    roc_auc_test = auc(fpr,tpr)



    print("Area Under Curve:   ",roc_auc_test)

    model.save('Subash_Chandra_Pakhrin'+str(seed)+str(Andrew)+'.h5')
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print()
    print('Subash_Chandra_Pakhrin'+str(seed)+str(Andrew)+'.h5')
    print()
    print('9999999999999999999999999999999')