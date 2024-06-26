import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import shap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class LearningAlgorithm:
    """
    A class implementing various machine learning algorithms and utilities.

    Attributes:
    -----------
    model : object
        The machine learning model to be trained and used.
    scaler : object
        The scaler object used for data preprocessing.
    device : str
        The device ('cuda' or 'cpu') for PyTorch operations based on availability.

    Methods:
    --------
    preprocess_data(X, y=None, scale_type='standard', test_size=0.2):
        Preprocesses input data X and optionally target data y.
    train_model(X_train, y_train, model_type='classification', **kwargs):
        Trains a machine learning model on the given training data.
    evaluate_model(X_test, y_test, metric='accuracy'):
        Evaluates the trained model on the test data using specified metric.
    save_model(file_path):
        Saves the trained model to a file.
    load_model(file_path, model_type='classification'):
        Loads a trained model from a file.
    incremental_learning(X_new, y_new):
        Performs incremental learning by updating the model with new data.
    hyperparameter_tuning(X, y, model_type='classification', search_type='grid', param_distributions=None, n_iter=10):
        Optimizes hyperparameters of the model using cross-validation.
    transfer_learning(base_model, new_data, new_labels, num_epochs=5):
        Performs transfer learning using a pre-trained base model.
    anomaly_detection(data, contamination=0.05):
        Detects anomalies in the input data.
    ensemble_learning(models, X_train, y_train, X_test, y_test, method='average'):
        Performs ensemble learning using multiple models.
    interpret_model(X):
        Interprets the model predictions to understand feature importance or contributions.
    visualize_data(X, y=None, method='pca'):
        Visualizes the data using dimensionality reduction techniques.
    """

    def __init__(self):
        """
        Initializes a LearningAlgorithm instance.

        Sets up model, scaler, and device attributes.
        """
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_data(self, X, y=None, scale_type='standard', test_size=0.2):
        """
        Preprocesses input data X and optionally target data y.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data to be preprocessed.
        y : numpy.ndarray, optional
            Target data for supervised learning tasks, by default None.
        scale_type : str, optional
            Type of scaling to apply ('standard' or 'minmax'), by default 'standard'.
        test_size : float, optional
            Proportion of the data to be used as test set if y is provided, by default 0.2.

        Returns:
        --------
        If y is provided:
        X_train : numpy.ndarray
            Preprocessed training data.
        X_test : numpy.ndarray
            Preprocessed test data.
        y_train : numpy.ndarray
            Training labels.
        y_test : numpy.ndarray
            Test labels.
        
        If y is not provided:
        X_scaled : numpy.ndarray
            Preprocessed data.
        """
        if scale_type == 'standard':
            self.scaler = StandardScaler()
        elif scale_type == 'minmax':
            self.scaler = MinMaxScaler()
        
        X_scaled = self.scaler.fit_transform(X)

        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            return X_scaled

    def train_model(self, X_train, y_train, model_type='classification', **kwargs):
        """
        Trains a machine learning model on the given training data.

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        model_type : str, optional
            Type of model ('classification', 'regression', 'svm', 'svr', 'neural_network'), by default 'classification'.
        **kwargs : keyword arguments
            Additional arguments specific to the chosen model.

        Raises:
        -------
        ValueError
            If an unsupported model type is provided.
        """
        if model_type == 'classification':
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == 'regression':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'svm':
            self.model = SVC(**kwargs)
        elif model_type == 'svr':
            self.model = SVR(**kwargs)
        elif model_type == 'neural_network':
            self.model = NeuralNetwork(input_dim=X_train.shape[1], output_dim=len(set(y_train)))
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            self.model.to(self.device)
            self.model.train()
            for epoch in range(10):
                running_loss = 0.0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch {epoch+1}/{10}, Loss: {running_loss/len(dataloader)}")
        else:
            raise ValueError("Unsupported model type.")
        
        if model_type != 'neural_network':
            self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test, metric='accuracy'):
        """
        Evaluates the trained model on the test data using specified metric.

        Parameters:
        -----------
        X_test : numpy.ndarray
            Test data.
        y_test : numpy.ndarray
            Test labels.
        metric : str, optional
            Evaluation metric ('accuracy', 'roc_auc', 'mean_absolute_error'), by default 'accuracy'.

        Returns:
        --------
        float or numpy.ndarray
            Evaluation score based on the chosen metric.
        """
        y_pred = self.model.predict(X_test)
        
        if metric == 'accuracy':
            return accuracy_score(y_test, y_pred)
        elif metric == 'roc_auc':
            return roc_auc_score(y_test, y_pred)
        elif metric == 'mean_absolute_error':
            return mean_absolute_error(y_test, y_pred)
        else:
            raise ValueError("Unsupported metric.")

    def save_model(self, file_path):
        """
        Saves the trained model to a file.

        Parameters:
        -----------
        file_path : str
            Path where the model should be saved.
        """
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path, model_type='classification'):
        """
        Loads a trained model from a file.

        Parameters:
        -----------
        file_path : str
            Path from where the model should be loaded.
        model_type : str, optional
            Type of model ('classification', 'neural_network'), by default 'classification'.
        """
        if model_type == 'neural_network':
            self.model = NeuralNetwork(input_dim=X_train.shape[1], output_dim=len(set(y_train)))
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()
        else:
            self.model = torch.load(file_path)
            self.model.to(self.device)

    def incremental_learning(self, X_new, y_new):
        """
        Performs incremental learning by updating the model with new data.

        Parameters:
        -----------
        X_new : numpy.ndarray
            New data for training or updating the model.
        y_new : numpy.ndarray
            New labels for training or updating the model.

        Raises:
        -------
        NotImplementedError
            If the model does not support incremental learning.
        """
        if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, RandomForestRegressor):
            self.model.fit(X_new, y_new)
        elif hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new, y_new)
        else:
            raise NotImplementedError("Model does not support incremental learning.")

    def hyperparameter_tuning(self, X, y, model_type='classification', search_type='grid', param_distributions=None, n_iter=10):
        """
        Optimizes hyperparameters of the model using cross-validation.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Training labels.
        model_type : str, optional
            Type of model ('classification', 'regression', 'neural_network'), by default 'classification'.
        search_type : str, optional
            Type of hyperparameter search ('grid' or 'random'), by default 'grid'.
        param_distributions : dict, optional
            Dictionary with parameters names (string) as keys and distributions or lists of parameters to try, by default None.
        n_iter : int, optional
            Number of parameter settings that are sampled, by default 10.
        """
        if model_type == 'classification':
            model = RandomForestClassifier()
        elif model_type == 'regression':
            model = RandomForestRegressor()
        elif model_type == 'neural_network':
            model = NeuralNetwork(input_dim=X.shape[1], output_dim=len(set(y)))
        else:
            raise ValueError("Unsupported model type.")
        
        if search_type == 'grid':
            search = GridSearchCV(model, param_distributions, cv=5)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=5)
        else:
            raise ValueError("Unsupported search type.")
        
        search.fit(X, y)
        self.model = search.best_estimator_

    def transfer_learning(self, base_model, new_data, new_labels, num_epochs=5):
        """
        Performs transfer learning using a pre-trained base model.

        Parameters:
        -----------
        base_model : torch.nn.Module
            Pre-trained base model for transfer learning.
        new_data : numpy.ndarray
            New training data for fine-tuning the base model.
        new_labels : numpy.ndarray
            New training labels for fine-tuning the base model.
        num_epochs : int, optional
            Number of epochs for training, by default 5.
        """
        base_model.fc = nn.Linear(base_model.fc.in_features, len(set(new_labels)))
        base_model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(base_model.parameters(), lr=0.001)

        dataset = TensorDataset(torch.tensor(new_data, dtype=torch.float32), torch.tensor(new_labels, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            base_model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = base_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
        
        self.model = base_model

    def anomaly_detection(self, data, contamination=0.05):
        """
        Detects anomalies in the input data.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data to detect anomalies.
        contamination : float, optional
            Proportion of outliers in the data set, by default 0.05.

        Returns:
        --------
        numpy.ndarray
            Anomaly labels (-1 for outliers, 1 for inliers).
        """
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=contamination)
        clf.fit(data)
        return clf.predict(data)

    def ensemble_learning(self, models, X_train, y_train, X_test, y_test, method='average'):
        """
        Performs ensemble learning using multiple models.

        Parameters:
        -----------
        models : list
            List of machine learning models to be used for ensemble learning.
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        X_test : numpy.ndarray
            Test data.
        y_test : numpy.ndarray
            Test labels.
        method : str, optional
            Ensemble method ('average' or 'majority_vote'), by default 'average'.

        Returns:
        --------
        float
            Accuracy score of the ensemble model on the test data.
        """
        predictions = []
        for model in models:
            model.fit(X_train, y_train)
            predictions.append(model.predict(X_test))
        
        if method == 'average':
            final_predictions = np.mean(predictions, axis=0)
        elif method == 'majority_vote':
            from scipy.stats import mode
            final_predictions = mode(predictions, axis=0)[0]
        else:
            raise ValueError("Unsupported ensemble method.")
        
        return accuracy_score(y_test, final_predictions)

    def interpret_model(self, X):
        """
        Interprets the model predictions to understand feature importance or contributions.

        Parameters:
        -----------
        X : numpy.ndarray
            Data for which model interpretation is to be performed.

        Returns:
        --------
        numpy.ndarray
            SHAP values indicating feature importance or contributions.
        """
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        return shap_values

    def visualize_data(self, X, y=None, method='pca'):
        """
        Visualizes the data using dimensionality reduction techniques.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to be visualized.
        y : numpy.ndarray, optional
            Labels for data visualization, by default None.
        method : str, optional
            Dimensionality reduction technique ('pca' or 'tsne'), by default 'pca'.
        """
        if method == 'pca':
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            plt.scatter(components[:, 0], components[:, 1], c=y)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('PCA Visualization')
            plt.show()
        elif method == 'tsne':
            tsne = TSNE(n_components=2)
            components = tsne.fit_transform(X)
            plt.scatter(components[:, 0], components[:, 1], c=y)
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('t-SNE Visualization')
            plt.show()
        else:
            raise ValueError("Unsupported visualization method.")


class NeuralNetwork(nn.Module):
    """
    A simple neural network for classification tasks.

    Attributes:
    -----------
    fc1 : torch.nn.Linear
        First fully connected layer.
    fc2 : torch.nn.Linear
        Second fully connected layer.
    relu : torch.nn.ReLU
        ReLU activation function.
    softmax : torch.nn.Softmax
        Softmax activation function.

    Methods:
    --------
    forward(x):
        Forward pass of the neural network.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initializes a NeuralNetwork instance.

        Parameters:
        -----------
        input_dim : int
            Dimensionality of input features.
        output_dim : int
            Number of output classes for classification.
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor for the neural network.

        Returns:
        --------
        torch.Tensor
            Output tensor after passing through the network layers.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Example Usage
if __name__ == "__main__":
    la = LearningAlgorithm()

    # Example data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = la.preprocess_data(X_train, y_train)

    # Train different models
    la.train_model(X_train_scaled, y_train_scaled, model_type='classification')
    # la.train_model(X_train_scaled, y_train_scaled, model_type='neural_network')

    # Evaluate model
    accuracy = la.evaluate_model(X_test_scaled, y_test_scaled, metric='accuracy')
    roc_auc = la.evaluate_model(X_test_scaled, y_test_scaled, metric='roc_auc')
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC Score: {roc_auc}")

    # Save and load model
    la.save_model('model.pth')
    la.load_model('model.pth')

    # Hyperparameter tuning
    param_grid = {'n_estimators': [50, 100, 150]}
    la.hyperparameter_tuning(X_train_scaled, y_train_scaled, param_distributions=param_grid, search_type='grid')

    # Transfer learning
    from torchvision import models
    base_model = models.resnet18(pretrained=True)
    la.transfer_learning(base_model, X_train_scaled, y_train_scaled)

    # Anomaly detection
    anomalies = la.anomaly_detection(X)
    print(f"Anomalies detected: {anomalies}")

    # Ensemble learning
    models = [RandomForestClassifier(n_estimators=50), SVC(probability=True)]
    ensemble_accuracy = la.ensemble_learning(models, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    print(f"Ensemble Model Accuracy: {ensemble_accuracy}")

    # Interpret model (SHAP)
    shap_values = la.interpret_model(X_test_scaled)
    shap.summary_plot(shap_values, X_test_scaled, feature_names=None)

    # Visualize data
    la.visualize_data(X_train_scaled, y_train_scaled, method='tsne')
