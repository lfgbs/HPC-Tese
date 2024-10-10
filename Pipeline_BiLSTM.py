# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
from csv import DictWriter
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split,  TimeSeriesSplit, HalvingGridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, LayerNormalization, Input, Bidirectional
from keras import regularizers, saving
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from scikeras.wrappers import KerasRegressor

from skopt import BayesSearchCV

import tensorflow as tf
# -

# # Set seaborn style

sns.set_style('darkgrid')
sns.set_palette('colorblind', color_codes=True)
sns.set_context('paper')

# # Parameter Definition

# +
rivers=['Vouga', 'Mondego', 'Antua', 'Neiva']

scenarios={'Univariate':['RD'], 'Multivariate':['Prec','Temp'], 'Multivariate_nrp':['Prec','nr_prec','Temp']}

n_days_to_analyze = [1, 5, 20, 35]
n_days_to_predict = 1
nr_precipitation=30

split_sizes={'Antua':0.5, 'Neiva':0.5, 'Vouga':0.5, 'Mondego':0.5}

sensor_fault={'Prec':[0, np.inf], 'Temp':[-20,60], 'RD':[0, np.inf]}

min_folds=4
max_folds=6
batch_size=32
epochs=500
patience=5

hyperband=False
max_depth=3


# -

# ### Enginner nr_prec and Add Memory

def compute_nrprec(features, nr_precipitation):

    #pad dataset so there is no information loss after shift
    features.loc[max(features.index)+nr_precipitation, :] = None

    #shift the data by nr_precipitation rows
    shifted_prec=features['Prec'].shift(nr_precipitation)
    
    nr_prec=[]

    #for each data point, slice the previous nr_precipitation days. Append NaN or avg of the slice to nr_prec, according to wether or not there were enough preceding days to calculate the average 
    for n in range(features.shape[0]):
        nr_prec.append( np.nan if any(np.isnan(shifted_prec.iloc[n:n+nr_precipitation])) else np.sum(shifted_prec.iloc[n:n+nr_precipitation])/nr_precipitation)
    
    return nr_prec


def add_memory(features, labels, desired_columns, n_in=1, n_out=1): 

    f, f_columns, l, l_columns = list(), list(), list(), list()  #inicialize empty lists

    features=features[desired_columns]
    
    f_names=list(features.columns)
    l_name=list(labels.columns)

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        f.append(features.shift(i))
        f_columns += [("{}-{}".format(name, i)) for name in f_names]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        l.append(labels.shift(-i))
        l_columns += [("{}+{}".format(l_name,i))]
            
    ## put it all together
    agg_features = pd.concat(f, axis=1)
    agg_features.columns=f_columns
    
    agg_labels = pd.concat(l, axis=1)
    agg_labels.columns=l_columns

    #DROP NA VALUES
    agg_features.dropna(inplace=True)
    agg_labels.dropna(inplace=True)
    #get indexes common to both lists
    common_indexes=np.intersect1d(agg_features.index, agg_labels.index)
    #keep only the common indexes
    #use df.loc instead of iloc because we are referring to the index labels, not the index themselves
    agg_labels = agg_labels.loc[common_indexes]
    agg_features=agg_features.loc[common_indexes]

    
    return agg_features, agg_labels


# ### Cut by batch_size and Compute number of folds

class TSSMismatch(Exception):

    def __init__(self, min_folds, max_folds):
        self.min_folds = min_folds
        self.max_folds = max_folds

    def __str__(self):
        return "Set does not allow for the creation of folds in the range of [{}, {}] ".format(self.min_folds, self.max_folds)


def ensure_TSS(n_batches, min_folds, max_folds):

    if n_batches>=min_folds:
        for folds in range(min_folds, max_folds+1):
            if n_batches%folds==0:
                return n_batches, folds
        return ensure_TSS(n_batches-1, min_folds, max_folds)
    
    raise TSSMismatch(min_folds, max_folds)


def cut_by_batch_size(X, y, batch_size, train=False, min_folds=np.inf, max_folds=np.inf):

    #dummy value for folds in case train=False
    folds=0
    
    #cut training data
    n_train_batches=len(X)//batch_size

    #when dealing with training data we must ensure tss
    if train:
        n_train_batches, folds= ensure_TSS(n_train_batches, min_folds, max_folds)

    X_cut=X[:batch_size*n_train_batches]
    y_cut=y[:batch_size*n_train_batches]

    return X_cut, y_cut, folds


# # Create Pipeline steps

# ### checking for Sensor faults in RD, Temp and Prec

class OutliersToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, col_bounds=None):
        self.col_bounds = col_bounds
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        for column, bounds in self.col_bounds.items():

            #variables have a suffix pertaining to the memory. we search the dataframe to retrieve all that start with the given prefix
            affected_columns=list(X.filter(like=column).columns)
            
            not_faulty=(X[affected_columns]>=bounds[0]) & (X[affected_columns]<=bounds[1])
            X[affected_columns]=X[affected_columns].where(not_faulty, np.nan)            
        
        return X


# ### Reshape LSTM input

class ReshapeInput(BaseEstimator, TransformerMixin):
    def __init__(self, memory, n_features):
        self.memory = memory
        self.n_features=n_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_reshaped=X.reshape((X.shape[0], self.memory, self.n_features))
        
        return X_reshaped


class Debug(BaseEstimator, TransformerMixin):

    def __init__(self,stage):
        self.stage=stage

    def transform(self, X):
        print('Shape at {}: {}'.format(self.stage,X.shape))

        if self.stage=='end':
            print('--------------------------')

        return X

    def fit(self, X, y=None, **fit_params):
        return self


# +

#OutliersToNan - sensor_faults
outlier=OutliersToNaN(sensor_fault)
        
#Imputer
imputer=KNNImputer(n_neighbors=2, weights='distance')
        
#scaler
scaler = MinMaxScaler(feature_range=(0, 1))
        
#debuggers
debugger_outlier=Debug('outlier')
debugger_imputer=Debug('imputer')
debugger_scaler=Debug('scaler')
debugger_reshaper=Debug('reshaper')
debugger_end=Debug('end')


# -

# ### Create model

def build_model(depth, units, dropout_rate, batch_size, memory, n_features, l1, l2, lr):   
    model = Sequential() 

    model.add(Input(batch_shape=(batch_size, memory, n_features)))
    
    for i in range(depth):
        model.add(Bidirectional(LSTM(units=int(units/(2**i)), stateful=True, kernel_regularizer=None if i>0 else regularizers.L1L2(l1=l1, l2=l2), recurrent_dropout=dropout_rate, return_sequences=True if i<depth-1 else False)))
        model.add(LayerNormalization(rms_scaling=True)) 
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

    return model


# ### Search and Fit

class EpochEndReset(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                try:
                    layer.reset_states()
                except:
                    layer.forward_layer.reset_states()
                    layer.backward_layer.reset_states()



# +
es = EarlyStopping(monitor='loss', restore_best_weights=True, patience=patience)

for river in rivers:
    print("\tAnalyzing river {}".format(river))
        
    data = pd.read_csv('RD_{}R_pg.csv'.format(river))
        
    labels=data[['RD']]
    #drop Date
    data=data.drop(['Date'], axis=1)
    
    train_size=split_sizes[river]
    test_size=1-train_size
            
    #split data
    features_train, features_test, labels_train, labels_test=train_test_split(data, labels, train_size=train_size, shuffle=False)
            
    for days_of_memory in n_days_to_analyze:
            
        print("\t\tDays of Memory:{}".format(days_of_memory))
            
        #add nr_prec
        features_train['nr_prec']=compute_nrprec(features_train, nr_precipitation)
        features_test['nr_prec']=compute_nrprec(features_test, nr_precipitation)
                
        for scenario, desired_columns in scenarios.items():
                
            print("\t\t\tWorking on scenario {}".format(scenario))
                    
            #add memory
            X_train_uncut, y_train_uncut=add_memory(features_train, labels_train, desired_columns, days_of_memory, n_days_to_predict)
            X_test_uncut, y_test_uncut=add_memory(features_test, labels_test, desired_columns, days_of_memory, n_days_to_predict)
    
            #cut to match all batch sizes
            X_train, y_train, folds = cut_by_batch_size(X_train_uncut, y_train_uncut, batch_size, train=True, min_folds=min_folds, max_folds=max_folds)
            X_test, y_test, _ = cut_by_batch_size(X_test_uncut, y_test_uncut, batch_size) 
    
            #reset indexes for the purpose of plotting the data in the correct position
            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)
                                                               
            #Set up reshaper (n_samples, memory, n_features)
            reshaper=ReshapeInput(memory=days_of_memory, n_features=len(desired_columns))
    
            #set up instance
            instance=KerasRegressor(model=build_model, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es, EpochEndReset()])
                    
            #Set up Pipeline - When using Randomized/GridSearch and the scikeras wrapper I prefer to use Pipeline instead of pipeline because the name is important to assign the parameters to the model    
            #pipeline=Pipeline(steps=[('d1', debugger_outlier),('outlier', outlier),('d2', debugger_imputer),('imputer', imputer),('d3', debugger_scaler),('scaler', scaler),('d4', debugger_reshaper),('reshaper', reshaper),('d5',debugger_end),('instance', instance)])
            pipeline=Pipeline(steps=[('outlier', outlier),('imputer', imputer),('scaler', scaler),('reshaper', reshaper),('instance', instance)])
        
            #To check how the params should be referenced
            #print("\n",pipeline.get_params().keys())
    
            tss=TimeSeriesSplit(folds-1)
    
            #hyperparameters to search
            param_grid={'instance__model__units':[32,64,128],'instance__model__dropout_rate':[0, 0.2, 0.4], 
                        'instance__model__l1':[0, 0.01, 0.1, 1], 'instance__model__l2':[0, 0.01, 0.1, 1], 
                        'instance__model__lr':[0.001, 0.01, 0.1], 'instance__model__depth':list(range(1, max_depth+1)), 
                        'instance__model__batch_size':[batch_size], 'instance__model__memory':[days_of_memory], 'instance__model__n_features':[len(desired_columns)]
                       } 
    
           if halving:
                print("Running HalvingGridSearchCV")
    
                trials="exhaustive"
    
                tuner=HalvingGridSearchCV(pipeline, param_grid=param_grid, verbose=1, cv=tss, n_jobs=-1)
                
            else:
                print("Running BayesSearchCV")
                
                trials=int(np.floor(max(0.15, np.log10(max_depth))*40))
                
                #Set up BayesSearchCV
                tuner=BayesSearchCV(pipeline, search_spaces=param_grid, verbose=1, cv=tss, random_state=40, n_iter=trials, n_jobs=1)
                
            search_start_time=perf_counter()
            tuner.fit(X_train, y_train)
            search_end_time=perf_counter()
                    
            y_pred_test=tuner.predict(X_test)
            y_pred_train=tuner.predict(X_train)
    
            #plot train predictions over real data
            ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
            ax1.plot(y_train,color='k')
            ax1.plot(y_pred_train,color='g')
            ax1.tick_params(labelbottom=False, labelleft=False)
    
            #plot test predictions over real data
            ax2 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
            ax2.plot(y_test,color='k')
            ax2.plot(y_pred_test,color='r')
            ax2.tick_params(labelbottom=False, labelleft=False)
    
            #plot all predictions over real data
            all_data=np.concatenate((y_train, y_test))
            ax3 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            ax3.plot(all_data, color='k', label='Real data')
            ax3.plot(y_pred_train,color='g', label='Train predictions')
            ax3.plot(list(range(y_pred_train.shape[0], all_data.shape[0])), y_pred_test, color='r', label='Test predictions')
    
            #display legend
            ax3.legend(loc="upper right", fontsize='xx-small')
    
            #set figure title
            plt.gcf().suptitle("{} river RD predictions\n {}".format(river ,scenario))
    
            #save figure
            plt.savefig('images/plots/BiLSTM/{}_{}_{}daysmemory.eps'.format(river, scenario, days_of_memory), format='eps', dpi=100)
                
            # Show the plot
            plt.close()
                    
            rmse_train = root_mean_squared_error(y_train, y_pred_train)
            rmse_test = root_mean_squared_error(y_test, y_pred_test)
            mae_train=mean_absolute_error(y_train, y_pred_train)
            mae_test=mean_absolute_error(y_test, y_pred_test)
            r2_train=r2_score(y_train, y_pred_train)
            r2_test=r2_score(y_test, y_pred_test)
    
            k=days_of_memory*len(desired_columns)
            
            adj_r2_train = 1- ((1-r2_train) * (y_pred_train.shape[0]-1)/(y_pred_train.shape[0]-k-1))
            adj_r2_test = 1- ((1-r2_test) * (y_pred_test.shape[0]-1)/(y_pred_test.shape[0]-k-1))
    
            fit_times=tuner.cv_results_['mean_fit_time']
            mean_fit_time=np.sum(fit_times)/len(fit_times)
            print("\t\t\t\t\tMean fit time: {}".format(mean_fit_time))
        
            bp=tuner.best_params_
        
            new_row = {'river':river,'scenario':scenario, 'model':'BiLSTM','stateful':'True', 'split':'TimeSeriesSplit', 'folds':folds, 'trials':trials,
                       'train_set_size':train_size,'n_samples_train': X_train.shape[0], 'test_set_size':test_size, 'n_samples_test': X_test.shape[0], 'memory': days_of_memory,
                       'r2_train': round(r2_train,3), 'r2_test': round(r2_test,3), 'adj_r2_train': round(adj_r2_train,3), 'adj_r2_test': round(adj_r2_test,3),
                       'rmse_train': round(rmse_train, 3), 'rmse_test': round(rmse_test, 3), 
                       'mae_train': round(mae_train,3), 'mae_test': round(mae_test, 3), 'loss':'mse', 
                       'alpha': None, 'batch_size':batch_size, 'epochs':epochs, 'early_stopping': patience, 'depth':bp['instance__model__depth'],'units':bp['instance__model__units'], 
                        'dense_dropout_rate':bp['instance__model__dropout_rate'], 'l1':bp['instance__model__l1'], 'l2':bp['instance__model__l2'], 'learning_rate':bp['instance__model__lr'],
                        'kernel_size':None, 'pool_size':None, 'activation':None, 'padding':None, 'training_time': round(mean_fit_time,3), 'search_time':round(search_end_time-search_start_time,3)
                        }
        
            file_path = os.path.join(os.getcwd(), 'pipeline.csv') 
        
            try:
                with open(file_path, 'a') as csv:
                    dictwriter_object = DictWriter(csv, fieldnames=list(new_row.keys()) )
                    dictwriter_object.writerow(new_row)
                    csv.close()
                
            except OSError as error:
                print("Error")
                    
            print('-------------------------------------------------------------------------')
# -


