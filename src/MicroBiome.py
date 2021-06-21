import numpy as np
import pandas as pd
from sklearn import model_selection
import copy
from vizwiz import VizWiz
from scipy.stats import ttest_ind as ttest
from statsmodels.stats.multitest import fdrcorrection
from sklearn.pipeline import Pipeline

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

class DiffExpTransform():
    """
    Apply Differential Abundance Analysis to 2 or more groups. Transformer-like object

    attributes:
        selected_feats = indicator vector for features selected for further
        results = dict of differential abundance results. T-test is performed comparing in class samples to
            out of class samples. T-statistic will be positive if mean of in-class set is greater than mean
            of out of class set
        fdr = false discovery rate
    """
    def __init__(self, fdr=0.05):
        self.selected_feats = None
        self.results = {}
        self.fdr = fdr
        self.expected_shape = None

    def check_X_(self, X):

        try:
            assert len(X.shape) == 2
        except:
            raise ValueError("expect X to be 2D array. X has shape {}".format(X.shape))

        if not self.expected_shape is None:
            try:
                assert X.shape == self.expected_shape
            except:
                raise ValueError("Expect X to have shape {}, input has shape {}".format(self.expected_shape, X.shape))

    def check_Y_(self, y):

        try:
            assert len(y.shape) == 1
        except:
            raise ValueError("expect y to be 1D array. y has shape {}".format(y.shape))

        try:
            assert (np.issubdtype(y.dtype, np.integer) || np.issubdtype(y.dtype, np.str_))
        except:
            raise ValueError("expect y to be integer or string")

    def fit(self, X, y):
        """

        :param X: m x n matrix of values
        :param y: vector of length m, denoting 2 or more classes
        :return: object is modified with selected features
        """

        self.check_X_(X)
        self.check_Y_(y)
        y_classes = np.sort(np.unique(y))
        self.selected_feats = np.zeros((X.shape[1],)).astype('bool')

        for c in y_classes:
            in_class = np.in1d(y, c)
            X1 = X[in_class, :]
            X2 = X[np.logical_not(in_class), :]
            tstat, pval = ttest(X1, X2, axis=0, alternative='greater')
            rejected, p_adj = fdrcorrection(pval, alpha=self.fdr)
            result_df = pd.DataFrame({tstat: tstat, pval: pval, p_adj: p_adj, rejected: rejected})
            self.results[str(c)] = result_df
            self.selected_feats[np.logical_not(rejected)] = True

    def transform(self, X):
        """

        :param X: m x n matrix of values
        :return: X, subsetted for only features of interest
        """

        if self.expected_shape is None:
            raise ValueError('self.expected_shape is None, run fit method before calling transform')

        self.check_X_(X)
        X_subs = X[:, self.selected_feats]
        return X_subs

            
class list_transformer():
    """
    apply pipelines to list
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms
        
    def check(self, X):
        if not isinstance(X, list):
            raise TypeError('X must be a list')
        if not len(self.transforms) == len(X):
            raise ValueError('X must have same number of elements as transforms')
            
    def fit(self, X):
        
        for i in range(len(self.transforms)):
            self.transforms[i].fit(X[i])
            
    def transform(self, X):
        """
        Applies transformations to all elements in X
        Assumes that all transforms output matrices
        """
        
        X_cat = None
        
        for i in range(len(self.transforms)):
            if i == 0:
                X_cat = self.transforms[i].transform(X[i])
            else:
                X_cat = np.concatenate((X_cat, X[i]), axis=1)
                
        return X_cat
            
        
# class transformer_X():
#     """
#     Transform a list of microbiome and clinical data
#     """
#     def __init__(self, use_PCA=True, n_components=30):
#         self.microbiome_scaler = StandardScaler()
#         self.microbiome_PCA = PCA(n_components=n_components)
#         self.PCA_scaler = StandardScaler()
#         self.clinical_scaler = StandardScaler()
#         self.use_PCA = use_PCA
        
#     def fit(self, DataList):
#         """
#         params:
#             DataList = list of 3 arrays, 1 (index 0) expected to be microbiome data, 2 (index 1) expected to be continuous
#             clinical data, 3 (index 2) expected to be binary clinical data. 
#         Note:
#             You should sort all clinical data by binary status if you would like the order of varianbles preserved for clinical
#             data. If PCA is run, first n_components variables are standard normalized PCA values, all variables after that are
#             from elements 2 and 3 of DataList.
#         """
#         MBiomeX = DataList[0]
#         ClinicalContinousX = DataList[1]
#         ClinicalBinaryX = DataList[2]
#         MBiomeX = self.microbiome_scaler.fit_transform(MBiomeX)
        
#         if self.use_PCA:
#             MBiomeX = self.microbiome_PCA.fit_transform(MBiomeX)
#             MBiomeX = self.microbiome_PCA.fit_transform(MBiomeX)
            
#         if not ClinicalContinousX is None:
#             ClinicalContinousX = self.clinical_scaler.fit_transform(ClinicalContinousX)
#             XFinal = np.concatenate((MBiomeX, ClinicalContinousX), axis=1)
#         else:
#             XFinal = MBiomeX
        
#     def transform(self, DataList):
#         """
#         params:
#             DataList = list of 3 arrays, 1 (index 0) expected to be microbiome data, 2 (index 1) expected to be continuous
#             clinical data, 3 (index 2) expected to be binary clinical data. 
#         Note:
#             You should sort all clinical data by binary status if you would like the order of varianbles preserved for clinical
#             data. If PCA is run, first n_components variables are standard normalized PCA values, all variables after that are
#             from elements 2 and 3 of DataList.
#         """
        
#         MBiomeX = DataList[0]
#         ClinicalContinousX = DataList[1]
#         ClinicalBinaryX = DataList[2]
        
#         MBiomeX = self.microbiome_scaler.transform(MBiomeX)
        
#         if self.use_PCA:
#             MBiomeX = self.microbiome_PCA.transform(MBiomeX)
            
#         if not ClinicalContinousX is None:
#             ClinicalContinousX = self.clinical_scaler.transform(ClinicalContinousX)
#             XFinal = np.concatenate((MBiomeX, ClinicalContinousX), axis=1)
#         else:
#             XFinal = MBiomeX
        
#         if not ClinicalBinaryX is None:
#             XFinal = np.concatenate((XFinal, ClinicalBinaryX), axis=1)
        
#         return XFinal
        
        
class Trainer:
    """
    Wrapper for training allowing for std normalization and/or PCA prior to running
    
    model = sklearn model implementing fit and predict
    transformer
    use_pca = use PCA on X prior to feeding to model
    n_components = # components for PCA
    """
    
    def __init__(self, model, pipeline_X = None, pipeline_y = None, **args):
        
        self.model = model
        self.pipeline_X = pipeline_X
        self.pipeline_y = pipeline_y
        
        
    def transform_X(self, X):
        
        if not self.pipeline_X is None:
            X = self.pipeline_X.transform(X)
            
        return X
    
    def transform_y(self, y):
        
        if not self.pipeline_y is None:
            y = self.pipeline_y.transform(y)
            
        return y 
        
    def fit(self, X, y, epochs=1, class_weight={}, batch_size=1, validation_data=None):
        # print("ARGS")
        # print(**args)
        if not self.pipeline_X is None:
            self.pipeline_X.fit(X)
            
        if not self.pipeline_y is None:
            self.pipeline_y.fit(y)
            
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
        score_value = score(y, y_pred)
        return score_value
        

class TrainTester:
    

    def __init__(self, TrainerObj, score, test_frac = 0.20, rand_state = 42, use_proba_predict=True, **args):
        """
        TrainerObj = object of Trainer class
        score = score function from sklearn.metrics, of signature score(y_true, y_predicted)
        test_frac = fraction to be held out for testing
        rand_state = exactly what it sounds like
        use_proba_predict = use class probability estimates to make final estimates.
        """
        self.Trainer = TrainerObj
        self.score_use = score
        self.test_frac = test_frac
        self.rand_state = rand_state
        self.use_proba_predict = use_proba_predict
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.y_train = None
        self.y_test = None
        self.y_train_pred_proba = None
        self.y_test_pred_proba = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.train_score = None
        self.test_score = None
        self.history = None
        
    def train(self, X, y, do_validation=False, **args):
        
        train_idx, test_idx = model_selection.train_test_split(np.arange(0, y.shape[0]), 
                                                                random_state = self.rand_state, 
                                                                test_size = self.test_frac)
        if do_validation:
            train_idx, val_idx = model_selection.train_test_split(train_idx, 
                                                                random_state = self.rand_state, 
                                                                test_size = self.test_frac)
        else:
            val_idx = None
            
        if isinstance(X, list):
            X_train = [ data_i[train_idx] for data_i in X ]
            if do_validation:
                X_val = [ data_i[val_idx] for data_i in X ]
            X_test = [ data_i[test_idx] for data_i in X ]
        else:
            X_train = X[train_idx, :]
            if do_validation:
                X_val = X[val_idx, :]
            X_test = X[test_idx, :]
        
        y_train = y[train_idx]
        if do_validation:
            y_val = y[val_idx]
        y_test = y[test_idx]
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.X_train = X_train
        self.X_test = X_test
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
            if self.use_proba_predict:
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
        self.train_score = self.Trainer.score(X_train, y_train, self.score_use)
        self.test_score = self.Trainer.score(X_test, y_test, self.score_use)
        

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
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []
        self.X_train = []
        self.X_test = []
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
            self.train_idx.append( TrainTesterCopy.train_idx )
            self.val_idx.append( TrainTesterCopy.val_idx )
            self.test_idx.append( TrainTesterCopy.test_idx )
            self.X_train.append( TrainTesterCopy.X_train )
            self.X_test.append( TrainTesterCopy.X_test )
            self.y_train.append( TrainTesterCopy.y_train )
            self.y_test.append( TrainTesterCopy.y_test )
            self.y_train_pred_proba.append( TrainTesterCopy.y_train_pred_proba )
            self.y_test_pred_proba.append( TrainTesterCopy.y_test_pred_proba )
            self.y_train_pred.append( TrainTesterCopy.y_train_pred )
            self.y_test_pred.append( TrainTesterCopy.y_test_pred )
            self.history.append( TrainTesterCopy.history )
            
            
        