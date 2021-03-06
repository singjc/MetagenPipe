import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import datetime



class FeedForward(nn.Module):
    """
    Feed Forward neural network,
    """
    def __init__(self, inp_size, hidden_layer_sizes):
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

    def forward(self, X):
        """
        Forward Pass
        :param X: mxn matrix
        :return: mxp matrix, where p is the output size of the last hidden layer
        """
        for fc in self.layers:
            X = F.relu(fc(X))

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
        :param size: size of random sample to take. Sampling done with replacement
        :param return_encoded: If True, return encoded data, if false, return non-encoded data
        """

        if not X.shape[0] == y.shape[0]:
            raise ValueError('number of samples in X ' + X.shape[0] + ' does not match number of samples in y ' + y.shape[0])
        m = X.shape[0]
        self.size = size
        self.X = X
        self.y = y
        self.encoder = encoder
        self.return_encoded = return_encoded
        np.random.seed(rand_seed)
        all_inds = np.random.randint(0, m, 2*size)
        # randomly chosen indexes for X1 and X2
        self.X1_inds = all_inds[0:size]
        self.X2_inds = all_inds[size:len(all_inds)]
        # samples of X given randomly chosen indices
        self.X1 = self.X[self.X1_inds, :].clone().detach()
        self.X2 = self.X[self.X2_inds, :].clone().detach()
        # vector c returns whether or not 2 class vectors corresponding to X1 and X2 are of the same class
        c = []
        self.c = torch.tensor([np.all(np.equal(self.y[self.X1_inds[i], :].numpy(), self.y[self.X2_inds[i], :].numpy())) for i in range(size)]).to(torch.float32)
        self.X1_out = None
        self.X2_out = None
#         print(self.X.dtype)
#         print(self.X1.dtype)
#         print(self.X2.dtype)
#         print(self.c.dtype)

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


class SiameseModel:
    """
    Siamese Neural Network, modeled off of sklearn model API
    """
    def __init__(self, model, update_encoder=True, predict_unknown=True, learning_rate=1e-3,weight_decay=1e-5,
                 batch_size=1000, num_epochs=5, rand_seed=42, class_min_train=10,
                 n_example_predict=20, k=5, max_dist=None, train_size=20000, validation_frac=0.10):
        """
        Initialize Object

        Note: Training step involves training on triplets and also calculating a validation loss based on validation triplets
                Prediction involves storing a database of class examples' encoded positions.
                If max_dist=None, max_dist automatically set to the minimum of the following 2 values:
                5th percentile of inter-class differences
                95th percentile of intra-class distances
                If median euclidean distance (in latent space) between query and members of assigned class is > max_dist,
                assign unknown label (0, 0, ..., 0, 0) for given sample. Note that this is currently a heuristic and
                needs to be evaluated in context of correctly predicting unseen classes.

                Use loss function as as defined by Koch, Zemel, Salakhutdinov
                in Siamese Neural Networks for One-shot Image Recognition
                Proceedings of the 32 nd International Conference on Machine
                Learning, Lille, France, 2015


        :param model: neural net outputting latent space
        :param update_encoder: update encoder, i.e. run siamese NN training. If set to False, skip siamese NN training
        when self.fit(X,y) called, and instead encode data into latent space and make predictions based on encoder's
        current values. False setting useful for assessing if latent space can distinguish unseen classes.
        :param predict_unknown: boolean. Whether or not to
        :param learning_rate: learning rate
        :param weight_decay: see docs for pytorch adam optimizer. L2 penalty on params
        :param batch_size: batch size for training + validation
        :param num_epochs: how many epochs to run training for
        :param rand_seed: random seed for splitting data, selecting examples for the predict function.
        :param class_min_train: minimum necessary examples necessary for training and validation sets
        :param n_example_predict: examples to select per class for the predict function
        :param k: number of nearest neighbors in KNN to predict classes.
        :param max_dist: maximum distance between 2 classes. If median distance to predicted class is > max_dist,
                query fed to .predict method is given 'unclassified' status. See note above for automatic determination
                of max_dist if max_dist set to None.
        :param train_size: number of training triplets to make. number of validation triplets is np.floor(train_size*(validation_frac)/(1-validation_frac)
        :param validation_frac: fraction of samples to use for validation. Data is split in stratified fashion. If None, do not do validation step.
        """

        self.model = model
        self.logistic = LinearSoftMax(inp_size=model.out_size*2, out_size=2)
        self.update_encoder = update_encoder
        self.predict_unknown = predict_unknown
        self.optimizer = torch.optim.Adam((list(self.model.parameters()) + list(self.logistic.parameters())), lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.rand_seed = rand_seed
        self.class_min_train = class_min_train
        self.n_example_predict = n_example_predict
        self.k = k
        self.max_dist = max_dist
        self.train_size = train_size
        self.validation_frac = validation_frac
        self.TrainData = None
        self.TrainDL = None
        self.ValData = None
        self.ValDL = None
        self.ClassDB = None
        
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
#         torch.manual_seed(self.rand_seed)
        # convert to float32 tensor
        X, y = self.__process_Xy(X, y)
        
        if self.update_encoder:
            self.__trainEncoder(X, y)

        self.__makeDataBase(X, y)


    def __trainEncoder(self, X, y):
        """
        :param X: mxn matrix
        :param y: mxc one-hot encoded class matrix
        :return: object with encoder in model attribute updated
        """

        if not self.validation_frac is None:
            self.__splitData(X, y)
        else:
            self.TrainData = self.__makeDS(X, y)
        
#         print(X.dtype)
#         print(y.dtype)
        for i in range(self.num_epochs):
            print('#########################################')
            print('Epoch {0} of {1}'.format(i + 1, self.num_epochs))
            print('__Training__')
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            mean_loss = self.run_epoch(mode='Train')
            print('MEAN LOSS: {}'.format(mean_loss))
            if not self.validation_frac is None:
                print('__Validation__')
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                mean_loss = self.run_epoch(mode='Val')
                print('MEAN LOSS: {}'.format(mean_loss))

        print('#########################################')
        print('Finished')
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))

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
        DL = DataLoader(DS, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        return DS, DL

    def __splitData(self, X, y):
        """
        Split Data into test and train fractions
        :return: object is modified with TrainData and ValData set
        """
        np.random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)
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
            X_cat = torch.cat([X1_encoded, X2_encoded], dim=1)
            logistic_output = self.logistic(X_cat)
            loss = loss_fun(logistic_output[:, 1], c)
            total_loss += loss.item()
            loss_batch_mean = loss.divide(float(len(c)))
            print('Batch Mean Loss: {}'.format(loss_batch_mean.item()))
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

    def pairwise_dists(self, X):
        """
        Compute euclidean distances between all vectors in X
        :param X:
        :return:
        """
        return_torch = False
        if isinstance(X, torch.Tensor):
            return_torch = True
            X = X.detach().numpy()

        dist_mat = pairwise_distances(X, metric = 'euclidean')
        if return_torch:
            dist_mat = torch.tensor(dist_mat)

        return dist_mat

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
            np.random.seed(self.rand_seed)
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

        dist = self.pairwise_dists(X_out_subs)
#         print(dist)
        # find 95th percentile of intraclass distances
        intraclass_dists = torch.tensor([])
#         print(dist.shape)
        for idx_class in class_inds_list:
#             print(idx_class.shape)
#             print(idx_class)
            dist_class = dist[:, idx_class]
            dist_class = dist_class[idx_class, :]
#             print(dist_class)
#             print(dist_class.shape)
            triu_class = torch.triu_indices(dist_class.shape[0], dist_class.shape[1], offset=1)
#             print(triu_class)
#             print(triu_class.shape)
            dist_class_values = torch.flatten(dist_class[triu_class[0, :], triu_class[1, :]])
            print(dist_class_values.shape)
            intraclass_dists = torch.cat((intraclass_dists, dist_class_values))
            
            
#         print(intraclass_dists)
#         print(intraclass_dists.detach().numpy())
        intraclass_95th = torch.tensor([np.percentile(intraclass_dists.detach().numpy(), 95)]).to(torch.float32)
        self.ClassDB['intraclass_dists'] = intraclass_dists

        # find 5th percentile of interclass dists
        interclass_dists = torch.tensor([])

        for i in range(len(class_inds_list) - 1):
            idx_i = class_inds_list[i]
            for j in range(i + 1, len(class_inds_list)):
                idx_j = class_inds_list[j]
                dists_i_j = dist[idx_i, :]
                dists_i_j = dist[:, idx_j]
                dists_i_j = torch.flatten(dists_i_j)
                interclass_dists = torch.cat((interclass_dists, dists_i_j))

        self.ClassDB['interclass_dists'] = interclass_dists
        interclass_5th = torch.tensor([np.percentile(intraclass_dists.detach().numpy(), 5)]).to(torch.float32)

        if self.max_dist is None:
            self.max_dist = torch.min(torch.cat((intraclass_95th, interclass_5th)))

    def predict(self, X):
        """
        Make prediction on X. NOTE:
        :param X:
        :return: matrix of shape (X.shape[0], n_classes) or (X.shape[0], n_classes + 1) if predict_unknown set to True

        """
        is_numpy = False
        is_tensor = False
        if isinstance(X, np.ndarray):
            is_numpy = True
            X = torch.from_numpy(X).to(torch.float32)
        elif isinstance(X, torch.Tensor):
            is_tensor = True
        else:
            raise TypeError('Expect 2D np.ndarray or tensor')

        X = self.model.forward(X)
        X = X.detach().numpy()

        X_DB = self.ClassDB['X_encoded'].detach().numpy()
        y_DB = self.ClassDB['y'].detach().numpy()

        n_samples = X.shape[0]
        n_classes = y_DB.shape[1]

        KNN_clf = KNeighborsClassifier(n_neighbors=self.k, metric='minkowski', p=2)
        KNN_clf.fit(X_DB, y_DB)
        y_pred = KNN_clf.predict(X)

        if self.predict_unknown:

            print('Classifying outlier samples as unknown. output will have 1 additional class column')

            is_unknown = np.repeat(0, n_samples)

            for i in range(n_samples):
                class_i = np.where(np.equal(y_pred[i, :], 1))[0]
                class_inds = np.where(np.equal(y_DB[:, class_i], 1))[0]
                diff = X[i,:] - X_DB[class_inds, :]
                class_dists = np.sqrt(np.diag(np.matmul(diff, diff.T)))
                median_dist = np.median(class_dists)

                if median_dist > self.max_dist:
                    y_pred[i, class_i] = 0
                    is_unknown[i] = 1
            # turn is_unknown into a column vector and append to y_pred
            is_unknown = is_unknown.reshape((len(is_unknown), 1))
#             print(y_pred.shape)
#             print(is_unknown.shape)
            y_pred = np.concatenate((y_pred, is_unknown), axis = 1)
            
        if is_tensor:
            # return tensor if tensor input. return numpy otherwise
            y_pred = torch.from_numpy(y_pred)

        return y_pred
    
    def predict_proba(self, X):
        """
        Make prediction on X. Uses KNN classifier.
        TODO: potentially replace with logistic regression based comparison of vectors
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
            is_tensor = True
        else:
            raise TypeError('Expect 2D np.ndarray or tensor')

        X = self.model.forward(X)
        X = X.detach().numpy()

        X_DB = self.ClassDB['X_encoded'].detach().numpy()
        y_DB = self.ClassDB['y'].detach().numpy()

        n_samples = X.shape[0]
        n_classes = y_DB.shape[1]

        KNN_clf = KNeighborsClassifier(n_neighbors=self.k, metric='minkowski', p=2)
        KNN_clf.fit(X_DB, y_DB)
        y_pred = KNN_clf.predict_proba(X)
            
        if is_tensor:
            # return tensor if tensor input. return numpy otherwise
            y_pred = torch.from_numpy(y_pred)

        return y_pred






