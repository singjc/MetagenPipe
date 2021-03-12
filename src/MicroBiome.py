import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.decomposition import PCA
import copy
from vizwiz import VizWiz

# Define class for indexing data
class MicroBiomeDataSet:
    """
    Container for matrix data, sample metadata, feature metadata
    """
    
    def __init__(self, data, sample_meta, feat_meta):
        """
        data = np.ndarray of data
        sample_meta = pandas DataFrame with each row corresponding to a sample
        feat_meta = pandas DataFrame with each row corresponding to a feature
        """
        if sample_meta.shape[0] != data.shape[0]:
            raise ValueError('expect # rows in sample_meta to match # rows of data')
            
        if feat_meta.shape[0] != data.shape[1]:
            raise ValueError('expect # rows in feat_meta to match # columns of data')
        self.data = data
        self.sample_meta = sample_meta
        self.feat_meta = feat_meta
        
        
    def idx_data(self, rows = None, cols = None, slot = 'data', df_cols = None):
        """
        rows = data rows to be indexed. ndarray of integers or strings
        cols = data cols to be indexed. ndarray of integers or strings
        slot = attribute to pull from. one of 'data', 'sample_meta', 'feat_meta'
        df_cols = DataFrame cols if slot is 'sample_meta' or 'feat_meta'
        returns: data matrix with selected rows and cols, or sample or feature metadata with selected df_cols and
        selected dataframe rows (data rows for sample metadata, data cols for feature metadata)
        """
        
        if rows is None:
            # by default take all samples
            rows = np.arange(0, self.data.shape[0])
            
        if cols is None:
            # by default take all features
            cols = np.arange(0, self.data.shape[0])
            
        def str_to_idx(arr, df, rows_cols = 'rows'):
            """
            internal function to retrieve index given string + data frame
            arr = array
            df = dataframe
            rows_cols = rows -> compare array to rownames (df.index.values) of df
                        cols -> compare array to colnames (np.array)
            """
            if rows_cols == 'rows':
                idx_use = df.index.values
            elif rows_cols == 'cols':
                idx_use = df.keys().to_numpy()
            else:
                raise ValueError('Not implemented for value of rows_cols: ' + rows_cols)
                
            if not np.all(np.in1d(arr, idx_values)):
                raise ValueError('not all values in array in dataframe indices')
                
            idx_return = np.in1d(df.index.values, arr)
            return(idx_return)
            
        if issubclass(np.dtype(rows), np.str):
            # convert sample string ids to integer index
            rows = str_to_idx(rows, self.sample_meta, 'rows')          
        if issubclass(np.dtype(cols), np.str):
            # convert feature string ids to integer index
            cols = str_to_idx(cols, self.feat_meta, 'rows')
            
        # placeholder value for item_return
        item_return = None
        
        if slot == 'data':
            # return subset of data
            item_return = self.data[rows, cols]
            
        else:
            # return subset of one of the metadata dataframes
            if slot == 'sample_meta':
                df_use = self.sample_meta
                idx_use = rows
                
            elif slot == 'feat_meta':
                df_use = self.feat_meta
                idx_use == cols
                
            else:
                raise ValueError('Not implemented for value of slot: ' + slot)
            
            if df_cols is None:
                # by default use all columns
                df_cols = np.arange(0, df_use.shape[1])
                
            elif issubclass(np.dtype(df_cols), np.str):
                df_cols = str_to_idx(df_cols, df_use, 'cols')
                
            item_return = df_use.iloc[idx_use, df_cols]
            
        if item_return is None:
            raise ValueError('item_return should not be None')
        
        return item_return
            
        
        
class Trainer:
    """
    Wrapper for training allowing for std normalization and/or PCA prior to running
    
    model = sklearn model implementing fit and predict
    scale_X = standard normalize X
    scale_y = standard normalize y
    use_pca = use PCA on X prior to feeding to model
    n_components = # components for PCA
    """
    
    def __init__(self, model, scale_X = True, scale_y = False, use_pca = False, n_components = 100, **args):
        
        self.model = model
        self.X_scaler = None
        self.y_scaler = None
        
        if scale_X:
            self.X_scaler = StandardScaler()
            
        if scale_y:
            self.y_scaler = StandardScaler()
            
        self.pca_model = None
        
        if use_pca:
            self.pca_model = PCA(n_components = n_components)
        
    def transform_X(self, X):
        
        if not self.X_scaler is None:
            X = self.X_scaler.transform(X)
            
        if not self.pca_model is None:
            X = self.pca_model.transform(X)
            
        return X
    
    def transform_y(self, y):
        
        if not self.y_scaler is None:
            y = self.y_scaler.transform(y)
            
        return y 
        
    def fit(self, X, y, epochs=1, class_weight={}, batch_size=1, validation_data=None):
        # print("ARGS")
        # print(**args)
        if not self.X_scaler is None:
            self.X_scaler.fit(X)
            
        if not self.y_scaler is None:
            self.y_scaler.fit(y)
            
        if not self.pca_model is None:
            self.pca_model.fit(X)
            
        X = self.transform_X(X)
        y = self.transform_y(y)    
        if validation_data is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, epochs=epochs, class_weight=class_weight, batch_size=batch_size, validation_data=validation_data)
        
    def predict(self, X):
        
        X = self.transform_X(X)
        y_pred = self.model.predict(X)
        return(y_pred)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            X = self.transform_X(X)
            y_pred_proba = self.model.predict_proba(X)
            return(y_pred_proba)
        else:
            return None

        
    def score(self, X, y, score):
        """
        X = input, untransformed
        y = output, untransformed
        score = score from sklearn metrics. e.g. balanced accuracy
        """
        y_pred = self.predict(X)
        y = self.transform_y(y)
        score_value = score(y_pred, y)
        

class TrainTester:
    
    def __init__(self, TrainerObj, score, test_frac = 0.20, rand_state = 42, **args):
        """
        TrainerObj = object of Trainer class
        score = score function from sklearn.metrics, of signature score(y_true, y_predicted)
        test_frac = fraction to be held out for testing
        """
        self.Trainer = TrainerObj
        self.score = score
        self.test_frac = test_frac
        self.rand_state = rand_state
        self.y_train = None
        self.y_test = None
        self.y_train_pred_proba = None
        self.y_test_pred_proba = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.train_score = None
        self.test_score = None
        self.history = None
        
    def train(self, X, y, use_indices=False, do_validation=False, **args):
        
        if use_indices:
            X_train_idx, X_test_idx, y_train, y_test = model_selection.train_test_split(range(0, X[0].shape[0]), y, 
                                                                                random_state = self.rand_state, 
                                                                                test_size = self.test_frac)
            if do_validation:
                X_train_idx, X_val_idx, y_train, y_val = model_selection.train_test_split(range(0, len(X_train_idx)), y_train, 
                                                                                random_state = self.rand_state, 
                                                                                test_size = self.test_frac)
            X_train = [ data_i[X_train_idx] for data_i in X ]
            X_val = [ data_i[X_val_idx] for data_i in X ]
            X_test = [ data_i[X_test_idx] for data_i in X ]
        else:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                                random_state = self.rand_state, 
                                                                                test_size = self.test_frac)
        self.y_train = y_train 
        self.y_test = y_test 
        if do_validation:
            history = self.Trainer.fit(X_train, y_train, validation_data=[X_val,y_val], **args)
            self.history = history
        else:
            self.Trainer.fit(X_train, y_train, **args)

        if hasattr(self.Trainer, 'predict_proba'):
            print("Using predict_proba")
            y_train_pred_proba = self.Trainer.predict_proba(X_train)
            y_test_pred_proba = self.Trainer.predict_proba(X_test)
            self.y_train_pred_proba = y_train_pred_proba
            self.y_test_pred_proba = y_test_pred_proba
            # y_train_pred = np.where(y_train_pred_proba > 0.5, 1, 0)
            # y_test_pred = np.where(y_test_pred_proba > 0.5, 1, 0)
            if self.y_train_pred_proba is not None:
                print("getting predictions from probs")
                y_train_pred = np.argmax(y_train_pred_proba, axis=1)
                y_test_pred = np.argmax(y_test_pred_proba, axis=1)
            else: 
                print("getting predictions from predict")
                y_train_pred = self.Trainer.predict(X_train)
                y_test_pred = self.Trainer.predict(X_test)
                y_train_pred = np.where(y_train_pred > 0.5, 1, 0)
                y_test_pred = np.where(y_test_pred > 0.5, 1, 0)
        else:
            y_train_pred = self.Trainer.predict(X_train)
            y_test_pred = self.Trainer.predict(X_test)
        self.y_train_pred = y_train_pred 
        self.y_test_pred = y_test_pred 
        # print(f"y_train:{y_train_pred}")
        self.train_score = self.score(y_train_pred, y_train)
        self.test_score = self.score(y_test_pred, y_test)
        

class MultiTrainTester(VizWiz):
    
    def __init__(self, TrainTester, n_splits = 5, numpy_rand_seed = 42, **args):
        """
        TrainTester = TrainTester, to be deep copied for n_splits splits
        n_splits = number of splits
        numpy_rand_seed = seed used to generate seed sequence for train test split random state
        """
        self.template = TrainTester
        self.n_splits = n_splits
        self.rand_seed = numpy_rand_seed
        self.TrainerList = []
        self.y_train = []
        self.y_test = []
        self.y_train_pred_proba = []
        self.y_test_pred_proba = []
        self.y_train_pred = []
        self.y_test_pred = []
        self.train_scores = []
        self.test_scores = []
        self.seeds = None
        self.history = []

    def buildEncoder(self, classLabels):
        """
            Build and Encoder mapping
            
            Constructs a dictionary that maps each class label to a list 
            where one entry in the list is 1 and the remainder are 0
        """
        encodingLength = len(classLabels)
        encoder = {}
        mapper = {}
        for i, label in enumerate(classLabels):
            encoding = [0] * encodingLength
            encoding[i] = 1
            encoder[label] = encoding
            mapper[label] = i
        return encoder, mapper
    
    def train(self, X, y, **args):
        
        np.random.seed(self.rand_seed)
        seed_sequence = np.random.rand(self.n_splits)*100
        seed_sequence = seed_sequence.astype('int64')
        self.seeds = seed_sequence
        
        for i in range(self.n_splits):
            print('Running for split ' + str(i + 1) + ' of ' + str(self.n_splits))
            TrainTesterCopy = copy.deepcopy(self.template)
            TrainTesterCopy.rand_state = seed_sequence[i]
            TrainTesterCopy.train(X, y, **args)
            self.train_scores.append(TrainTesterCopy.train_score)
            self.test_scores.append(TrainTesterCopy.test_score)
            self.TrainerList.append(TrainTesterCopy.Trainer)
            self.y_train.append( TrainTesterCopy.y_train )
            self.y_test.append( TrainTesterCopy.y_test )
            self.y_train_pred_proba.append( TrainTesterCopy.y_train_pred_proba )
            self.y_test_pred_proba.append( TrainTesterCopy.y_test_pred_proba )
            self.y_train_pred.append( TrainTesterCopy.y_train_pred )
            self.y_test_pred.append( TrainTesterCopy.y_test_pred )
            self.history.append( TrainTesterCopy.history )
            
            
        