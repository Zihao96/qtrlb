import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from torch.utils.data import Dataset, DataLoader
from qtrlb.processing.plotting import COLOR_LIST



class Dataset(Dataset):
    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor):
        self.tensors = tensors
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index) -> tuple[torch.Tensor]:
        return self.tensors[index], self.labels[index]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, layer_stack: torch.nn.Sequential):
        super().__init__()
        self.layer_stack = layer_stack


    def forward(self, x):
        """
        Perform forward calculation.
        This function will be called when we called the instance of our model.
        But we shouldn't directly use it since there are more things in __call__.
        """
        logits = self.layer_stack(x)
        return logits


class ThreeLayersWithReLU(NeuralNetwork):
    """arXiv:2407.17407"""
    def __init__(self, n_features: int, n_labels: int):
        shrink_ratio = np.cbrt(n_features / n_labels)
        super().__init__(
            torch.nn.Sequential(
                torch.nn.Linear(n_features, round(n_labels*shrink_ratio**2)),
                torch.nn.ReLU(),
                torch.nn.Linear(round(n_labels*shrink_ratio**2), round(n_labels*shrink_ratio)),
                torch.nn.ReLU(),
                torch.nn.Linear(round(n_labels*shrink_ratio), n_labels),
            )
        )


class ModelTraining:
    """
    A general class for training a model. It's typically used for final training.
    It accepts model, loss and optimizer as types for the convenience of cross validations.
    """
    def __init__(self, 
                 device: str,
                 Model: type, 
                 Loss_fn: type, 
                 Optimizer: type,
                 train_data_tensor: torch.Tensor, 
                 train_label_tensor: torch.Tensor,
                 batch_size: int,
                 epochs: int,
                 **kwargs):
        self.device = device
        self.Model = Model
        self.Loss_fn = Loss_fn
        self.Optimizer = Optimizer
        self.train_data_tensor = train_data_tensor
        self.train_label_tensor = train_label_tensor
        self.batch_size = batch_size
        self.epochs = epochs
        for key, value in kwargs.items(): setattr(self, key, value)


    def run(self, model_kwargs: dict = None, loss_fn_kwargs: dict = None, optimizer_kwargs: dict = None):
        """
        Run the training process. This class and this method is typically used for final training.
        """
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.loss_fn_kwargs = {} if loss_fn_kwargs is None else loss_fn_kwargs
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        
        self.get_dataloader()
        self.generate_model()
        self.generate_loss_fn()
        self.generate_optimizer()
        self.train_model()


    def get_dataloader(self):
        """
        Get the training dataloader.
        """
        self.train_dataloader = DataLoader(
            Dataset(self.train_data_tensor, self.train_label_tensor), 
            self.batch_size, 
            shuffle=True
        )


    def generate_model(self):
        """
        Generate a self.model.
        """
        self.model = self.Model(**self.model_kwargs)


    def generate_loss_fn(self):
        """
        Generate a self.loss_fn.
        """
        self.loss_fn = self.Loss_fn(**self.loss_fn_kwargs)


    def generate_optimizer(self):
        """
        Generate a self.optimizer.
        """
        self.optimizer = self.Optimizer(self.model.parameters(), **self.optimizer_kwargs)


    def train_model(self):
        """
        Train the model with several epochs without any validation set.
        """
        metrics = {m: [] for m in ('train_loss', 'train_accr', 'valid_loss', 'valid_accr')}

        # Take the metrics before training
        self.validate(metrics, valid_dataloader=self.train_dataloader)
        metrics['train_loss'].append(metrics['valid_loss'].pop())
        metrics['train_accr'].append(metrics['valid_accr'].pop())

        # Training and real time plotting
        fig, ax = plt.subplots(1, 1, dpi=150)

        for epoch in range(1, self.epochs+1):
            # Train and validate
            self.train(metrics)
            x = np.arange(0, epoch+1, 1, dtype=int)

            # Refresh figure
            ax.cla()
            for m, values in metrics.items(): 
                if m.startswith('train'): ax.plot(x, values, label=m)
            ax.set(title=f'Loss and accuracy of final training',
                   xlabel='Epochs', ylabel='loss/accuracy', xlim=(0, self.epochs))
            ax.legend()
            ax.grid()
            display.display(fig)
            display.clear_output(wait=True)

        self.metrics = metrics


    def train(self, metrics: dict = None):
        """
        Training for a single epoch.
        """
        # Set the model to training mode - important for batch normalization and dropout layers
        self.model.train()
        train_loss, train_accr = 0, 0

        for X, y in self.train_dataloader:
            # Forward calculation of loss
            # Ref on process y label
            # https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
            X = X.to(self.device)
            y = y.type(torch.LongTensor).to(self.device)
            self.model = self.model.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # pred and y here are also tensor. Loss is just scalar.

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            train_accr += (pred.argmax(1) == y).type(torch.float).sum().item()
            # Believe or not, the argmax just doesn't work on my M1 Mac.

        train_loss /= len(self.train_dataloader)  # Number of batches
        train_accr /= len(self.train_dataloader.dataset)  # Number of data, gives normalized shot of correct prediction.

        if isinstance(metrics, dict):
            metrics['train_loss'].append(train_loss)
            metrics['train_accr'].append(train_accr)


    def validate(self, metrics: dict = None, valid_dataloader: DataLoader = None):
        """
        Validation for a single epoch.
        """
        if valid_dataloader is None:
            try:
                valid_dataloader = self.valid_dataloader
            except AttributeError as e:
                e.add_note('No valid_dataloader found. Please provide a valid_dataloader.')
                raise e

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        self.model.eval()
        valid_loss, valid_accr = 0, 0

        # Evaluating the model with torch.no_grad() reduce unnecessary gradient computations 
        # and memory usage for tensors with requires_grad=True。
        with torch.no_grad():
            for X, y in valid_dataloader:
                # Forward calculation of loss
                X = X.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                self.model = self.model.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                valid_loss += loss.item()
                valid_accr += (pred.argmax(1) == y).type(torch.float).sum().item()

        valid_loss /= len(valid_dataloader)
        valid_accr /= len(valid_dataloader.dataset)

        if isinstance(metrics, dict):
            metrics['valid_loss'].append(valid_loss)
            metrics['valid_accr'].append(valid_accr)


    def save_metrics(self, filename: str = 'metrics.json'):
        """
        Save the metrics to a json file with given path and filename.
        """
        if not hasattr(self, 'metrics'):
            raise AttributeError('No metrics found. Please run the self.train/self.validate first.')
        
        with open(filename, 'w') as file:
            json.dump(self.metrics, file)


class KFoldCrossValidation(ModelTraining):
    def run(self, k: int, model_kwargs: dict = None, loss_fn_kwargs: dict = None, optimizer_kwargs: dict = None):
        """
        Run the k-fold cross validation.
        Attributes like 'i', 'model', 'loss_fn', 'optimizer', 'train_dataloader', 'valid_dataloader', \
        'metrics' will be reset during loop.
        """
        self.k = k
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.loss_fn_kwargs = {} if loss_fn_kwargs is None else loss_fn_kwargs
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

        if self.k <= 1: raise ValueError('k must be greater than 1.')
        self.metrics = []

        for i in range(self.k):
            self.i = i
            self.get_single_fold_dataloader()
            self.generate_model()
            self.generate_loss_fn()
            self.generate_optimizer()
            self.single_fold_train()
        self.plot_valid_accr_mean()


    def get_single_fold_dataloader(self):
        """
        Get the training and validation dataloader for a single fold.
        """
        X_train, y_train, X_valid, y_valid = self.get_k_fold_data()
        self.train_dataloader = DataLoader(Dataset(X_train, y_train), self.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(Dataset(X_valid, y_valid), self.batch_size, shuffle=True)


    def get_k_fold_data(self):
        """ Partially stealed from Internet. """
        fold_size = self.train_data_tensor.shape[0] // self.k
        X_train, y_train = None, None

        for j in range(self.k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = self.train_data_tensor[idx, :], self.train_label_tensor[idx]

            # Pick the i-th fold as validation data
            if j == self.i:
                X_valid, y_valid = X_part, y_part

            # Other data will become train data
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)

        return X_train, y_train, X_valid, y_valid


    def single_fold_train(self):
        """
        Run the training and validation for a single fold.
        """
        metrics = {m: [] for m in ('train_loss', 'train_accr', 'valid_loss', 'valid_accr')}

        # Take the metrics before training
        self.validate(metrics, valid_dataloader=self.train_dataloader)
        metrics['train_loss'].append(metrics['valid_loss'].pop())
        metrics['train_accr'].append(metrics['valid_accr'].pop())
        self.validate(metrics)

        # Training and real time plotting
        fig, ax = plt.subplots(1, 1, dpi=150)
        ax2 = ax.twinx()
        valid_accr = 0

        for epoch in range(1, self.epochs+1):
            # Train and validate
            self.train(metrics)
            self.validate(metrics)
            x = np.arange(0, epoch+1, 1, dtype=int)
            if metrics["valid_accr"][-1] > valid_accr: valid_accr = metrics["valid_accr"][-1]

            # Refresh figure
            ax.cla()
            ax2.cla()
            ax.plot(x, metrics['train_loss'], color=COLOR_LIST[0], label='train_loss')
            ax.plot(x, metrics['valid_loss'], color=COLOR_LIST[1], label='valid_loss')
            ax2.plot(x, metrics['train_accr'], color=COLOR_LIST[2], label='train_accr')
            ax2.plot(x, metrics['valid_accr'], color=COLOR_LIST[3], label='valid_accr')
            ax.set(title=f'fold:{self.i}, valid_accr: {valid_accr:.4f}',
                   xlabel='Epochs', ylabel='loss/accuracy', xlim=(0, self.epochs))
            ax2.set(xlim=(0, self.epochs), ylim=(0, 1))
            ax2.legend()
            ax2.grid()
            display.display(fig)
            display.clear_output(wait=True)

        self.metrics.append(metrics)


    def plot_valid_accr_mean(self, ylim: tuple = (0.0, 1), text_loc: str = 'lower right'):
        """
        Plot the mean of validation accuracy of all folds.
        """
        self.figure, ax = plt.subplots(1, 1, dpi=150)
        y_mean = 0

        for i, metrics in enumerate(self.metrics):
            y = np.array(metrics['valid_accr'])
            y_mean += y
            ax.plot(np.arange(len(y)), y, label=f'Fold {i}', alpha=0.2)

        ax.plot(np.arange(len(y)), y_mean / self.k, label='Mean', color='black')
        ax.set(xlabel='Epochs', ylabel='Validation Accuracy', title='Validation Accuracy', 
               ylim=ylim, xlim=(0, self.epochs))
        ax.legend(loc=text_loc)
        ax.grid()