import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import datetime
import warnings


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class FeedForward(nn.Module):
    """
    Feed Forward neural network,
    """
    def __init__(self, inp_size, hidden_layer_sizes, activation='relu'):
        """

        :param inp_size: input size
        :param hidden_layer_sizes: Output sizes of hidden layers. Size of last hidden layer
        gives output size
        """
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList()
        self.inp_size = inp_size
        self.out_size = hidden_layer_sizes[len(hidden_layer_sizes) - 1]
        fc_in_size = inp_size
        for s in hidden_layer_sizes:
            fc_s = nn.Linear(in_features=fc_in_size, out_features=s).to(torch.float32)
            self.layers.append(fc_s)
            fc_in_size = s
            
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise ValueError('Not implemented for activation {}'.format(activation))

    def forward(self, X):
        """
        Forward Pass
        :param X: mxn matrix
        :return: mxp matrix, where p is the output size of the last hidden layer
        """
        for fc in self.layers:
            X = self.activation(fc(X))

        return X

class LinearSoftMax(nn.Module):

    def __init__(self, inp_size, out_size = 2):
        super(LinearSoftMax, self).__init__()
        self.fc1 = nn.Linear(in_features=inp_size, out_features=out_size).to(torch.float32)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):

        X = self.fc1(X)
        X = self.softmax(X)
        return X


class SiameseDataSet(Dataset):
    """
    Dataset for training a siamese neural net
    """
    def __init__(self, X, y, encoder, size=20000, return_encoded=False, rand_seed=42):
        """
        Initialization.
        :param X: mxn matrix of vectors. a 2D tensor
        :param y: one-hot encoded matrix of classes. 2D tensor
        :param encoder: Encoder function. should be of class nn.Module
        :param size: dataset size
        :param return_encoded: If True, return encoded data, if false, return non-encoded data
        """

        if not isinstance(X, torch.Tensor):
            raise TypeError('X must be a tensor')
        if not isinstance(y, torch.Tensor):
            raise TypeError('y must be a tensor')

        if not X.shape[0] == y.shape[0]:
            raise ValueError('number of samples in X ' + X.shape[0] + ' does not match number of samples in y ' + y.shape[0])
        if not len(X.shape) == 2:
            raise ValueError('Expect X to be a 2D tensor')
        if not len(y.shape) == 2:
            raise ValueError('Expect y to be a 2D tensor')

        self.size = size
        self.X = X
        self.y = y
        self.X_use = None
        self.y_use = None
        self.encoder = encoder
        self.return_encoded = return_encoded
        seed_everything(rand_seed)

        # get class counts in y, make stacked matrices with balanced classes.
        # note that
        n_classes = y.shape[1]
        n_per_class = np.floor(size/(n_classes*2))
        class_counts = torch.sum(y, dim=0)
        X_stacked = None
        y_stacked = None

        # indices for X1 and X2
        X1_idx = np.array([])
        X2_idx = np.array([])
        class_idx_stacked_list = []

        m = 0

        # create stacked matrix of equal class composition
        for i in range(n_classes):
            # index class
            class_i_idx = torch.where(torch.eq(y_stacked[:, i], 1))
            # randomly index rows for given class, with replacement
            rand_inds = np.random.randint(0, class_counts[i].detach().numpy(), n_per_class)
            X_i = X[class_i_idx, :][rand_inds, :].clone.detach()
            # technically we could get away with just remaking the matrix y here, but not really
            # keen on respecifying matrix
            y_i = y[class_i_idx, :][rand_inds, :].clone.detach()
            if i == 0:
                X_stacked = X_i
                y_stacked = y_i
            else:
                X_stacked = torch.cat((X_stacked, X_i), dim=0)
                y_stacked = torch.cat((y_stacked, y_i), dim=0)

            # index for class samples in given index
            class_idx_stacked = np.arange(m, X_stacked.shape[0])
            class_idx_stacked_list.append(class_idx_stacked)
            X1_idx = np.concatenate((X1_idx, class_idx_stacked))
            class_idx_stacked_perm = np.random.choice(class_idx_stacked, size=len(class_idx_stacked), replace=False)
            X2_idx = np.concatenate((X2_idx, class_idx_stacked_perm))

            m = X_stacked.shape[0]

        # get indices necessary for out of class comparisons
        for i in range(n_classes):
            # pair each class's samples with randomly selected out of class samples
            in_class_idx = class_idx_stacked_list[i]
            out_class_idx = np.array([])

            for j in range(n_classes):
                if i == j:
                    next()
                else:
                    out_class_idx = np.concatenate(out_class_idx, class_idx_stacked[j])

            X1_idx = np.concatenate((X1_idx, in_class_idx))
            out_class_idx_rnd = np.random.choice(out_class_idx, size=in_class_idx.shape[0], replace=False)
            X2_idx = np.concatenate(X2_idx, out_class_idx_rnd)

        self.X_use = X_stacked
        self.y_use = y_stacked

        if X1_idx.shape[0] < size:
            size_diff = size - self.X1.shape[0]
            X1_idx_append = np.random.choice(X1_idx, size=size_diff, replace=False)
            X2_idx_append = np.random.choice(X2_idx, size=size_diff, replace=False)
            warnings.warn('Appending {} extra pairs to self.X1 and self.X2'.format(size_diff))
            X1_idx = np.concatenate((X1_idx, X1_idx_append))
            X2_idx = np.concatenate((X2_idx, X2_idx_append))
        elif X1_idx.shape[0] > size:
            raise ValueError('self.X1 should not have number of rows greater than size')

        # # samples of X given randomly chosen indices
        self.X1 = self.X_use[X1_idx, :].clone().detach()
        self.X2 = self.X_use[X2_idx, :].clone().detach()

        # vector c returns whether or not 2 class vectors corresponding to X1 and X2 are of the same class
        c = []
        for i in range(len(self.y_use)):
            c.append(np.all(np.equal(self.y_use[X1_idx[i], :].numpy(), self.y_use[self.X2_idx[i], :].numpy())))

        self.c = torch.tensor(c).to(torch.float32)
        self.X1_out = None
        self.X2_out = None

        try:
            assert self.X1.shape[1] == self.X2.shape[1]
            assert self.X1.shape[0] == size
            assert self.X1.shape[0] == self.X2.shape[0]
            assert self.X1.shape[0] == self.c.shape[0]
        except:
            print('self.X1 shape {} {}'.format(self.X1.shape[0], self.X1.shape[1]))
            print('self.X2 shape {} {}'.format(self.X2.shape[0], self.X2.shape[1]))
            print('self.c shape {} {}'.format(self.c.shape[0], self.c.shape[1]))
            raise ValueError('Incorrect sizing for self.X1, self.X2, or self.c')

    def run_forward(self):
        """
        Run forward pass of NN on X.
        :return: object is modified with self.X1_out, self.X2_out updated
        """
        self.X1_out = self.encoder.forward(self.X1)
        self.X2_out = self.encoder.forward(self.X2)

    def __getitem__(self, idx):
        """
        Get Item at specified index. Indexing goes through randomly sampled pairs for siamese nn training
        :param idx: integer index, between 0 and self.__len__() - 1
        :return: if return_encoded set to True, return appropriate indices of self.X1_out, self.X2_out,
        as well as self.y
        """

        if self.return_encoded:
            # return encoded vectors
            X1 = self.X1_out[idx, :]
            X2 = self.X2_out[idx, :]
        else:
            # return non-encoded vectors
            X1 = self.X1[idx, :]
            X2 = self.X2[idx, :]
        # class label(s)
        c = self.c[idx]

        # force X1 and X2 to be 2D tensors
        if len(X1.shape) == 1:
            X1 = torch.reshape(X1, (1, X1.shape[0]))
        if len(X2.shape) == 1:
            X2 = torch.reshape(X2, (1, X2.shape[0]))

        sample = {'X1': X1, 'X2': X2, 'c': c}
        return sample

    def __len__(self):
        return self.size


# def SiameseLoss(X1, X2, y_long):
#     """
#     Siamese loss function, as defined by Koch, Zemel, Salakhutdinov
#     in Siamese Neural Networks for One-shot Image Recognition
#     Proceedings of the 32 nd International Conference on Machine
#     Learning, Lille, France, 2015
#
#     :param X1: Tensor
#     :param X2: Tensor of identical dimensions as X
#     :param y_long: binarized vector representing whether or not samples in X1 and X2 for given row are of same class
#     :return: loss function value
#     """

def weight_init(m):
    """
    Initialize weights for linear layer
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ModuleList):
        pass
    elif isinstance(m, FeedForward):
        pass
    else:
        raise TypeError('Only implemented for linear layers or lists thereof. Type: {}'.format(type(m)))


class SiameseModel:
    """
    Siamese Neural Network, modeled off of sklearn model API
    """
    def __init__(self, model, reinit_weights=True, update_encoder=True, predict_unknown=True, proba_thresh=0.5, learning_rate=1e-3,weight_decay=1e-5,
                 batch_size=1000, num_epochs=5, rand_seed=42, class_min_train=10,
                 n_example_predict=20, train_size=20000, validation_frac=0.10):
        """
        Initialize Object

        Note: Training step involves training on triplets and also calculating a validation loss based on validation triplets
                Prediction involves storing a database of class examples' encoded positions, predicting the log probability
                of a match with examples of each class, and then selecting the class with highest median log-probability across
                stored database of examples

                Use loss function as as defined by Koch, Zemel, Salakhutdinov
                in Siamese Neural Networks for One-shot Image Recognition
                Proceedings of the 32 nd International Conference on Machine
                Learning, Lille, France, 2015


        :param model: neural net outputting latent space

        :param update_encoder: update encoder, i.e. run siamese NN training. If set to False, skip siamese NN training
        when self.fit(X,y) called, and instead encode data into latent space and make predictions based on encoder's
        current values. False setting useful for assessing if latent space can distinguish unseen classes.
        :param predict_unknown: boolean. Whether or not to return unknown status.
        :param proba_thresh: float. threshold probability at which to classify sample as an unknown.
        :param learning_rate: learning rate
        :param weight_decay: see docs for pytorch adam optimizer. L2 penalty on params
        :param batch_size: batch size for training + validation
        :param num_epochs: how many epochs to run training for
        :param rand_seed: random seed for splitting data, selecting examples for the predict function.
        :param class_min_train: minimum necessary examples necessary for training and validation sets
        :param n_example_predict: examples to select per class for the predict function
        :param train_size: number of training triplets to make. number of validation triplets is np.floor(train_size*(validation_frac)/(1-validation_frac)
        :param validation_frac: fraction of samples to use for validation. Data is split in stratified fashion. If None, do not do validation step.
        """
        
        seed_everything(rand_seed)
        
        self.model = model
        self.reinit_weights = reinit_weights
        self.logistic = LinearSoftMax(inp_size=model.out_size, out_size=2)
        self.update_encoder = update_encoder
        self.predict_unknown = predict_unknown
        self.proba_thresh = proba_thresh
        self.optimizer = torch.optim.Adam((list(self.model.parameters()) + list(self.logistic.parameters())), lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.rand_seed = rand_seed
        self.class_min_train = class_min_train
        self.n_example_predict = n_example_predict
        self.train_size = train_size
        self.validation_frac = validation_frac
        self.TrainData = None
        self.TrainDL = None
        self.ValData = None
        self.ValDL = None
        self.ClassDB = None
        self.one_hot = None
        self.TrainStats = None
        
    def __process_Xy(self, X, y):
        
        if not isinstance(X, torch.Tensor):
            try:
                X = torch.from_numpy(X)
            except:
                raise TypeError('X must be 2D pytorch tensor or numpy ndarray')
        if not isinstance(y, torch.Tensor):
            try:
                y = torch.from_numpy(y)
            except:
                raise TypeError('y must be 2D pytorch tensor or numpy ndarray')

        if not len(X.shape) == 2:
            raise ValueError('X must be 2D ndarray or tensor')

        if not len(y.shape) == 2:
            raise ValueError('y must be 2D ndarray or tensor')
        
        X = X.to(torch.float32)
        y = y.to(torch.int32)
        
        return X, y

    def fit(self, X, y):
        """
        Notes: first fits siamese NN
        :param X:
        :param y:
        :return: SiameseModel object is fit, with database of class examples set up.
        """
        
        seed_everything(self.rand_seed)
        if self.reinit_weights:
            # initializes weights after a seed given
            self.model.apply(weight_init)
        
#         torch.manual_seed(self.rand_seed)
        # convert to float32 tensor
        X, y = self.__process_Xy(X, y)
        
        if self.update_encoder:
            self.__trainEncoder(X, y)

        self.__makeDataBase(X, y)
        self.RefitOneHot()

    def RefitOneHot(self):
        """
        Refit One Hot Encoder
        :return: modify one-hot encoder
        """
        try:
            y_db = self.ClassDB['y']
            n_classes = y_db.shape[1]
        except:
            raise ValueError('Likely that self.ClassDB not set. Did you run self.fit method?')
        finally:
            pass

        if self.predict_unknown:
            n_classes += 1

        a = np.arange(0, n_classes)
        a = a.reshape((a.shape[0], 1))
        self.one_hot = OneHotEncoder(sparse=False)
        self.one_hot.fit(a)

    def __trainEncoder(self, X, y):
        """
        :param X: mxn matrix
        :param y: mxc one-hot encoded class matrix
        :return: object with encoder in model attribute updated
        """
        epoch = []
        mean_loss_train_values = []
        mean_loss_val_values = []
        
        if not self.validation_frac is None:
            self.__splitData(X, y)
        else:
            self.TrainData = self.__makeDS(X, y)
        print('Training Encoder')
#         print(X.dtype)
#         print(y.dtype)
        for i in range(self.num_epochs):
            print('#########################################')
            print('Epoch {0} of {1}'.format(i + 1, self.num_epochs))
            epoch.append(i + 1)
            print('__Training__')
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            mean_loss_train = self.run_epoch(mode='Train')
            print('MEAN LOSS: {}'.format(mean_loss_train))
            mean_loss_train_values.append(mean_loss_train)
            if not self.validation_frac is None:
                print('__Validation__')
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                mean_loss_val = self.run_epoch(mode='Val')
                mean_loss_val_values.append(mean_loss_val)
                print('MEAN LOSS: {}'.format(mean_loss_val))

        print('#########################################')
        print('Finished')
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        
        df_dict = {}
        df_dict['epoch'] = epoch
        df_dict['mean_loss'] = mean_loss_train_values
        df_dict['stage'] = np.repeat('train', len(epoch))
        self.TrainStats = pd.DataFrame(df_dict)
        
        if len(mean_loss_val_values) == len(epoch):
            df_dict_add = {}
            df_dict_add['epoch'] = epoch
            df_dict_add['mean_loss'] = mean_loss_val_values
            df_dict_add['stage'] = np.repeat('val', len(epoch))
            df_add = pd.DataFrame(df_dict_add)
            self.TrainStats = self.TrainStats.append(df_add)  

    def __check_n_classes(self, y, thresh):
        # y is expected to be a tensor
        n_per_class = torch.sum(y, axis=0)
        lt_thresh = torch.less(n_per_class, thresh)
        if torch.any(lt_thresh):
            lt_inds = torch.where(lt_thresh)
            msg = 'following classes have less than required number of examples per class:'
            for ind in lt_inds:
                msg = msg + ' ' + str(ind)
            raise ValueError(msg)

    def __makeDS(self, X, y, size, rand_seed):
        """
        :param X: mxn matrix
        :param y: mxc one-hot encoded class matrix
        :return:
        """
        self.__check_n_classes(y, self.class_min_train)
        DS = SiameseDataSet(X, y, encoder=self.model, size=size, return_encoded=False, rand_seed=rand_seed)
#         torch.manual_seed(rand_seed)
        DL = DataLoader(DS, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        return DS, DL

    def __splitData(self, X, y):
        """
        Split Data into test and train fractions
        :return: object is modified with TrainData and ValData set
        """
#         np.random.seed(self.rand_seed)
#         torch.manual_seed(self.rand_seed)
        X_train, X_val, y_train, y_val = train_test_split(X.numpy(), y.numpy(), stratify=y.numpy())
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)
        train_size = self.train_size
        val_size = int(np.floor(train_size*(self.validation_frac/(1. - self.validation_frac))))
        TrainDS, TrainDL = self.__makeDS(X_train, y_train, size = train_size, rand_seed=self.rand_seed)
        self.TrainData = TrainDS
        self.TrainDL = TrainDL
        ValDS, ValDL = self.__makeDS(X_val, y_val, size=val_size, rand_seed=self.rand_seed)
        self.ValData = ValDS
        self.ValDL = ValDL

    def run_epoch(self, mode = 'Train'):
        """

        :param loader: Dataloader
        :param mode: Train for training data, val for validation data
        :return: average loss across epoch
        """

        if mode == 'Train':
            DS = self.TrainData
            DL = self.TrainDL
        elif mode == 'Val':
            DS = self.ValData
            DL = self.ValDL
        else:
            raise ValueError('mode must be either\'Train\' or \'Val\'')

        loss_fun = torch.nn.BCELoss(reduction = 'sum')
        total_loss = 0

        total_samples = len(DS)
        m = 0
        for batch_i in DL:
            X1 = torch.flatten(batch_i['X1'], start_dim = 1).detach()
            X2 = torch.flatten(batch_i['X2'], start_dim = 1).detach()
            X1_encoded = self.model.forward(X1)
            X2_encoded = self.model.forward(X2)
            c = batch_i['c']
            # note that logistic regression step implemented as a softmax on a linear layer outputting 2 features.
            # logistic function equivalent to softmax on 2 classes
            X_abs_diff = torch.abs(X1_encoded - X2_encoded)
            logistic_output = self.logistic(X_abs_diff)
            loss = loss_fun(logistic_output[:, 1], c)
            total_loss += loss.item()
            loss_batch_mean = loss.divide(float(len(c)))
#             print('Batch Mean Loss: {}'.format(loss_batch_mean.item()))
            if mode == 'Train':
                self.optimizer.zero_grad()
#                 if m == 0:
#                     do_retain_graph = True
#                 else:
#                     do_retain_graph = False
                loss_batch_mean.backward()
                self.optimizer.step()
                m = 1
            del loss
            del loss_batch_mean

        loss_total_mean = total_loss/(float(total_samples))
        return loss_total_mean

    # def pairwise_dists(self, X):
    #     """
    #     Compute euclidean distances between all vectors in X
    #     :param X:
    #     :return:
    #     """
    #     return_torch = False
    #     if isinstance(X, torch.Tensor):
    #         return_torch = True
    #         X = X.detach().numpy()
    #
    #     dist_mat = pairwise_distances(X, metric = 'euclidean')
    #     if return_torch:
    #         dist_mat = torch.tensor(dist_mat)
    #
    #     return dist_mat

    def __makeDataBase(self, X, y):
        """

        :param X: mxn matrix of samples x features
        :param y: mxp matrix of binarized class labels
        :return: fill selff.ClassDB slot to have encoded coordinates of X and corresponding class labels y
        """
        # checks if we have requisite examples per each class to make a prediction
        self.__check_n_classes(y, self.n_example_predict)
        class_inds = np.array([])
        class_inds_list = []
        m = 0
        for i in range(y.shape[1]):
            class_i_inds = np.where(np.equal(y[:,i], 1))[0]
#             print(class_i_inds[0].shape)
#             print(class_i_inds[1].shape)
#             np.random.seed(self.rand_seed)
            class_i_selected = np.random.choice(class_i_inds, size=self.n_example_predict)
            class_inds = np.concatenate((class_inds, class_i_selected))
            m_end = m + self.n_example_predict
            class_inds_list.append(torch.from_numpy(np.arange(m, m_end)))
            m = m_end

        X_subs = X[class_inds, :]
        y_subs = y[class_inds, :]
        
        X_out_subs = self.model.forward(X_subs)

        self.ClassDB = {}
        self.ClassDB['X_encoded'] = X_out_subs
        self.ClassDB['y'] = y_subs

    def predict(self, X):
        """
        Make prediction on X. NOTE:
        :param X:
        :return: matrix of shape (X.shape[0], n_classes) or (X.shape[0], n_classes + 1) if predict_unknown set to True

        """
        is_tensor = False
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(torch.float32)
        elif isinstance(X, torch.Tensor):
            is_tensor = True
        else:
            raise TypeError('Expect 2D np.ndarray or tensor')

        y_proba = self.predict_proba(X)
        y_pred = torch.zeros(y_proba.shape)
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            y_proba_i = y_proba[i, :]
            # note that ties here are handled by selecting the first class. In practice this should be
            # unlikely to be problematic due to use of floating point numbers in predicting probability
            # if we are predicting unknown labels, we handle cases where 1 or more other classes
            # have equal probability by assigning them an unknown label.
            class_i = torch.argmax(y_proba_i)
            y_pred[i, class_i] = 1

        n_samples = y_proba.shape[0]
        n_classes = y_pred.shape[1]

        if self.predict_unknown:

            print('Classifying outlier samples as unknown. output will have 1 additional class column')

            is_unknown = torch.zeros((n_samples, 1))

            for i in range(n_samples):
                class_i = torch.where(torch.eq(y_pred[i, :], 1))
                class_prob_i = y_proba[i, class_i]
                num_equal = torch.sum(torch.eq(y_proba, class_prob_i))
                # True if more than one class with probability equal to max class probability
                multi_label = num_equal > 1
                # True if class probability is below probability threshold
                below_thresh = class_prob_i < self.proba_thresh
                if multi_label or below_thresh:
                    y_pred[i, :] = 0
                    is_unknown[i, 0] = 1

            y_pred = torch.cat((y_pred, is_unknown), axis=1)
            
        if is_tensor:
            # return tensor if tensor input. return numpy otherwise
            y_pred = torch.from_numpy(y_pred)

        return y_pred
    
    def predict_proba(self, X):
        """
        Make prediction on X. Uses KNN classifier.
        to obtain SNN based probability
        :param X:
        :return: matrix of shape (X.shape[0], n_classes)

        """
        is_numpy = False
        is_tensor = False
        if isinstance(X, np.ndarray):
            is_numpy = True
            X = torch.from_numpy(X).to(torch.float32)
        elif isinstance(X, torch.Tensor):
            pass
        else:
            raise TypeError('Expect 2D np.ndarray or tensor')

        X = self.model.forward(X)
        X_db = self.ClassDB['X_encoded']
        y_db = self.ClassDB['y']
        n_classes = y_db.shape[1]
        n_samples = X.shape[0]
        y_pred_proba = torch.zeros((n_samples, n_classes))

        for i in range(n_samples):
            # calculate summed log probabilities for each class, comparing to each sample
            # log probability calculated with logistic function
            X_diff_i = torch.reshape(X[i, :], (1, X.shape[1])) - X_db
            logistic_input = torch.abs(X_diff_i)
            logistic_output = self.logistic.forward(logistic_input)
            class_probs = torch.zeros(n_classes)

            for j in range(n_classes):
                # y_db[:, j] is 1D so this is safe to do
                class_j_idx = torch.where(torch.eq(y_db[:, j], 1))[0]
                # VERY IMPORTANT: logistic output column 2 (index 1)
                # is what's used in the loss function for the binary cross entropy.
                # it's the predicted likelihood.
                class_j_probs = logistic_output[class_j_idx, 1]
                class_probs[j] = torch.median(class_j_probs)

            y_pred_proba[i, :] = class_probs
            
        if is_numpy:
            # return tensor if tensor input. return numpy otherwise
            y_pred_proba = y_pred_proba.detach().numpy()

        return y_pred_proba
