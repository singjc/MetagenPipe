import numpy as np
import pandas as pd
from sklearn import model_selection
import copy
from vizwiz import VizWiz
from ScoreFunctions import *
from scipy.stats import ttest_ind as ttest
from scipy.stats import pearsonr, spearmanr, fisher_exact
from statsmodels.stats.multitest import fdrcorrection

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
    
class StatsTransform():
    """
    Base class for statistics based transforms. 
    attributes:
        selected_feats = indicator vector for features selected for further use
        results = dict containing results of statistical analysis. 
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
                assert X.shape[1] == self.expected_shape[1]
            except:
                raise ValueError("Expect X to have {} features, input has shape {}".format(self.expected_shape[1], X.shape))
                
    def fit(self, X, y):
        """
        Any subclass should implement a method 'fit' where a null hypothesis is evaluated given the provided data,
        for each feature in X. 
        """
        pass
    

class DiffExpTransform(StatsTransform):
    """
    Apply Differential Abundance Analysis to 2 or more groups. Transformer-like object

    attributes:
        selected_feats = indicator vector for features selected for further use
        results = dict of differential abundance results. T-test is performed comparing in class samples to
            out of class samples. T-statistic will be positive if mean of in-class set is greater than mean
            of out of class set
        fdr = false discovery rate
    """
    def __init__(self, fdr=0.05):
        """
        initialize transform
        """
        super().__init__(fdr)
        
    def check_Y_(self, y):

        try:
            assert len(y.shape) == 1
        except:
            raise ValueError("expect y to be 1D array. y has shape {}".format(y.shape))

        try:
            assert (np.issubdtype(y.dtype, np.integer) | np.issubdtype(y.dtype, np.str_))
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
        try:
            assert X.shape[0] == len(y)
        except:
            raise ValueError('Expect X dim 1 to be equal to y length')
            
        y_classes = np.sort(np.unique(y))
        self.selected_feats = np.zeros((X.shape[1],)).astype('bool')

        for c in y_classes:
            in_class = np.in1d(y, c)
            X1 = X[in_class, :]
            X2 = X[np.logical_not(in_class), :]
            tstat, pval = ttest(X1, X2, axis=0, alternative='greater')
            rejected, p_adj = fdrcorrection(pval, alpha=self.fdr)
            result_df = pd.DataFrame({'tstat': tstat, 'pval': pval, 'p_adj': p_adj, 'rejected': rejected})
            self.results[str(c)] = result_df
            self.selected_feats[rejected] = True

        self.expected_shape = X.shape
        return self

    def transform(self, X):
        """

        :param X: m x n matrix of values
        :return: X, subsetted for only features of interest
        """

        if self.expected_shape is None:
            raise ValueError('self.expected_shape is None, run fit method before calling transform')
        elif np.all(np.logical_not(self.selected_feats)):
            raise ValueError('all entries in self.selected_feats false. possible that features are not differential between classes')

        self.check_X_(X)
        X_subs = X[:, self.selected_feats]
        return X_subs


class CorTransform(StatsTransform):
    """
    Apply Pearson or Spearman Correlation analysis comparing input variables to a target variable
    """
    
    def __init__(self, fdr=0.05, metric='pearson'):
        """
        initialize transform
        """
        super().__init__(fdr)
        if metric == 'pearson':
            self.cor_func = pearsonr
        elif metric == 'spearman':
            self.cor_func = spearmanr
        else:
            raise ValueError('metric must be one of {pearson, spearman}')
        
    def check_Y_(self, y):

        try:
            assert len(y.shape) == 1
        except:
            raise ValueError("expect y to be 1D array. y has shape {}".format(y.shape))

        try:
            assert (np.issubdtype(y.dtype, np.integer) | np.issubdtype(y.dtype, np.str_))
        except:
            raise ValueError("expect y to be integer or string")

    def fit(self, X, y, **kwargs):
        """

        :param X: m x n matrix of values
        :param y: vector of length m, denoting 2 or more classes
        :return: object is modified with selected features
        """

        self.check_X_(X)
        self.check_Y_(y)
        try:
            assert X.shape[0] == len(y)
        except:
            raise ValueError('Expect X dim 1 to be equal to y length')
            
        self.selected_feats = np.zeros((X.shape[1],)).astype('bool')
        rho = np.repeat(np.nan, len(y))
        pval = np.repeat(np.nan, len(y))
        for i in range(0, len(y)):
            rho_i, pval_i = self.cor_func(X[:, i].flatten(), y, **kwargs)
            rho[i] = rho_i
            pval[i] = pval_i
            
        rejected, p_adj = fdrcorrection(pval, alpha=self.fdr)
        result_df = pd.DataFrame({'r': rho, 'pval': pval, 'p_adj': p_adj, 'rejected': rejected})
        self.results['results'] = result_df
        self.selected_feats[rejected] = True
        self.expected_shape = X.shape
        return self

    def transform(self, X):
        """

        :param X: m x n matrix of values
        :return: X, subsetted for only features of interest
        """

        if self.expected_shape is None:
            raise ValueError('self.expected_shape is None, run fit method before calling transform')
        elif np.all(np.logical_not(self.selected_feats)):
            raise ValueError('all entries in self.selected_feats false. possible that features are not differential between classes')

        self.check_X_(X)
        X_subs = X[:, self.selected_feats]
        return X_subs
    
def fisher_test_vect(x1, x2, x1_pos=None, x2_pos=None, verbose=True, **kwargs):
    """
    Perform fisher's exact test on 2 vectors, expected to be binary
    
    Note: you can pass non-boolean arrays, including those with more than 2 categories, but a 'positive' class must be
    specified for the array
    
    :param x1: vector1, a numpy ndarray
    :param x2: vector 2, a numpy ndarray
    :param x1_pos: class to consider positive class in x1. Must not be None if x1 not boolean ndarray
    :param x2_pos: class to consider positive class in x2. Must not be None if x2 not boolean ndarray
    :param **kwargs: kwargs passed to fisher_exact
    :return: fisher's exact test result
    """
    def process_x(x, pos_val, name):
        if pos_val is not None:
            x = x == pos_val
        else:
            try:
                assert x.dtype is np.dtype('bool')
            except:
                raise ValueError('{} must have positive value specified if not bool'.format(name))
        return x
    x1 = process_x(x1, x1_pos, 'x1')
    x2 = process_x(x2, x2_pos, 'x2')
    TT = np.sum(np.logical_and(x1, x2).astype('int32'))
    TF = np.sum(np.logical_and(x1, np.logical_not(x2)).astype('int32'))
    FT = np.sum(np.logical_and(np.logical_not(x1), x2).astype('int32'))
    FF = np.sum(np.logical_and(np.logical_not(x1), np.logical_not(x2)).astype('int32'))
    cont_tab = np.array([[TT, TF], [FT, FF]])
    oddsratio, pval = fisher_exact(cont_tab, **kwargs)
    
    if verbose:
        print('contingency table')
        print(cont_tab)
        print('odds ratio: {}; pval {}'.format(oddsratio, pval))
    
    return cont_tab, oddsratio, pval
    
            
class list_transformer():
    """
    apply pipelines to list
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms
        
    def check(self, X, y=None):
        if not isinstance(X, list):
            raise TypeError('X must be a list')
        if not len(self.transforms) == len(X):
            raise ValueError('X must have same number of elements as transforms')
            
    def fit(self, X, y=None):
        # Note: same y is passed for various transforms in list
        self.check(X, y)
        for i in range(len(self.transforms)):
            if self.transforms[i] is not None:
                self.transforms[i].fit(X[i], y)
            else:
                continue
            
    def transform(self, X):
        """
        Applies transformations to all elements in X
        Assumes that all transforms output matrices
        """
        
        X_cat = None
        
        for i in range(len(self.transforms)):
            
            if self.transforms[i] is not None :
                X_transf = self.transforms[i].transform(X[i]).copy()
            else:
                X_transf = X[i]
            
            if i == 0:
                X_cat = X_transf
            else:
                X_cat = np.concatenate((X_cat, X_transf), axis=1)
                
        return X_cat
        
        
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
            self.pipeline_X.fit(X, y)
            
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

        
    def score(self, X, y, score, use_proba=False, pos_class=1):
        """
        X = input, untransformed
        y = output, untransformed
        score = score from sklearn metrics. e.g. balanced accuracy
        use_proba = use probability/confidence output (i.e. predict_proba)
        """
        if use_proba:
            # expect predict_proba to output nxc array of n samples, c columns of class probabilities
            y_pred = self.predict_proba(X)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, pos_class]
        else:
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
                # if self.y_train_pred_proba is not None:
                print("getting predictions from probs")
                y_train_pred = np.argmax(y_train_pred_proba, axis=1)
                y_test_pred = np.argmax(y_test_pred_proba, axis=1)
                # else:
                #     print("getting predictions from predict")
                #     y_train_pred = self.Trainer.predict(X_train)
                #     y_test_pred = self.Trainer.predict(X_test)
                #     y_train_pred = np.where(y_train_pred > 0.5, 1, 0)
                #     y_test_pred = np.where(y_test_pred > 0.5, 1, 0)
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
        # marks whether or not training completed successfully for all splits
        self.trained = False

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
        # mark that training complete for all splits
        self.trained = True

    def getScores(self):
        """
        Get a variety of scores on test data. See ScoreFunctions.py for scores used
        :return: dictionary of score metrics
        """

        try:
            assert self.trained
        except:
            raise ValueError('self.trained is False. did you run training?')

        score_metrics = {'split': [], 'score_type': [], 'value': []}
        score_dict = getScoreDict()
        for i in range(self.n_splits):
            X_test = self.X_test[i]
            y_test = self.y_test[i]
            Trainer_i = self.TrainerList[i]
            for score_key in score_dict.keys():
                score_entry = score_dict[score_key]
                score_func = score_entry['score_func']
                score_metrics['split'].append(i)
                score_metrics['score_type'].append(score_key)
                score_metrics['value'].append(Trainer_i.score(X_test, y_test, score_func, use_proba=score_entry['use_proba']))

        return score_metrics
