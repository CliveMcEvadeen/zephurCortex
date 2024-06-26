"""
utils.py

This module contains utility functions and classes used across the ZephyrCortex project.

Features:
- Logging Utility: Configures and provides logging functionalities.
- Configuration Loader: Loads configuration files.
- Data Loading Utility: Loads datasets from various sources.
- Metrics Calculation: Computes additional metrics.
- Plotting Utilities: Functions for plotting graphs.
- Data Serialization: Functions for saving and loading data to/from files.
- Environment Checker: Checks the computing environment.
- Timing Utility: Measure execution time of code blocks.
- Memory Usage: Check memory usage of objects.
- Model Serialization: Save and load machine learning models.
- Argument Parsing: Handle command-line arguments.
- Email Notifications: Send email notifications.

Dependencies:
- logging
- yaml
- json
- numpy
- pandas
- matplotlib
- seaborn
- torch
- joblib
- smtplib
- argparse

Example:
    from utils import setup_logging, load_config

    logger = setup_logging()
    config = load_config('config.yaml')
    
"""

import logging
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import time
import tracemalloc
import joblib
import smtplib
from email.mime.text import MIMEText
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import argparse


def setup_logging(log_file='project.log'):
    """
    Sets up logging configuration.

    Parameters:
    -----------
    log_file : str, optional
        Path to the log file, by default 'project.log'.

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    return logger


def load_config(config_path):
    """
    Loads a configuration file.

    Parameters:
    -----------
    config_path : str
        Path to the configuration file (YAML or JSON).

    Returns:
    --------
    config : dict
        Configuration parameters as a dictionary.
    """
    with open(config_path, 'r') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(file)
        elif config_path.endswith('.json'):
            config = json.load(file)
        else:
            raise ValueError("Unsupported configuration file format.")
    return config


def load_dataset(file_path):
    """
    Loads a dataset from a file.

    Parameters:
    -----------
    file_path : str
        Path to the dataset file (CSV or Excel).

    Returns:
    --------
    data : pandas.DataFrame
        Loaded dataset as a DataFrame.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format.")
    return data


def save_data(data, file_path):
    """
    Saves data to a file.

    Parameters:
    -----------
    data : pandas.DataFrame
        Data to be saved.
    file_path : str
        Path to the output file (CSV or Excel).
    """
    if file_path.endswith('.csv'):
        data.to_csv(file_path, index=False)
    elif file_path.endswith('.xlsx'):
        data.to_excel(file_path, index=False)
    else:
        raise ValueError("Unsupported file format.")


def calculate_metrics(y_true, y_pred, average='binary'):
    """
    Calculates various evaluation metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    average : str, optional
        Averaging method for multi-class data, by default 'binary'.

    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics.
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average=average),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average)
    }
    return metrics


def plot_roc_curve(y_true, y_pred):
    """
    Plots the ROC curve.

    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def check_environment():
    """
    Checks the computing environment for GPU availability.

    Returns:
    --------
    dict
        Dictionary indicating the availability of GPU.
    """
    env_info = {
        'gpu_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    }
    return env_info


def time_it(func):
    """
    A decorator to measure the execution time of a function.

    Parameters:
    -----------
    func : function
        Function to be timed.

    Returns:
    --------
    wrapper : function
        Wrapped function with timing.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper


def get_memory_usage():
    """
    Returns the current memory usage of the program.

    Returns:
    --------
    memory_usage : dict
        Dictionary containing memory usage information.
    """
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage = {
        'current': current / 10**6,  # Convert to MB
        'peak': peak / 10**6  # Convert to MB
    }
    return memory_usage


def save_model(model, file_path):
    """
    Saves a machine learning model to a file.

    Parameters:
    -----------
    model : object
        Machine learning model to be saved.
    file_path : str
        Path to the output file.
    """
    joblib.dump(model, file_path)


def load_model(file_path):
    """
    Loads a machine learning model from a file.

    Parameters:
    -----------
    file_path : str
        Path to the model file.

    Returns:
    --------
    model : object
        Loaded machine learning model.
    """
    return joblib.load(file_path)


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    --------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='ZephyrCortex Project Command Line Arguments')
    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    parser.add_argument('--log', type=str, help='Path to the log file.')
    args = parser.parse_args()
    return args


def send_email(subject, body, to, from_email, smtp_server, smtp_port, login, password):
    """
    Sends an email notification.

    Parameters:
    -----------
    subject : str
        Email subject.
    body : str
        Email body.
    to : str
        Recipient email address.
    from_email : str
        Sender email address.
    smtp_server : str
        SMTP server address.
    smtp_port : int
        SMTP server port.
    login : str
        SMTP login username.
    password : str
        SMTP login password.
    """
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(login, password)
        server.sendmail(from_email, [to], msg.as_string())


# Example Usage
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging is set up.")

    # Load configuration
    config = load_config('config.yaml')
    logger.info(f"Configuration loaded: {config}")

    # Load dataset
    data = load_dataset('data.csv')
    logger.info("Dataset loaded.")

    # Calculate metrics
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    metrics = calculate_metrics(y_true, y_pred)
    logger.info(f"Metrics calculated: {metrics}")

    # Plot ROC curve
    plot_roc_curve(y_true, y_pred)

    # Check environment
    env_info = check_environment()
    logger.info(f"Environment info: {env_info}")

    # Check memory usage
    memory_usage = get_memory_usage()
    logger.info(f"Memory usage: {memory_usage}")

    # Time a sample function
    @time_it
    def sample_function():
        time.sleep(2)
    sample_function()

    # Save and load model
    model = {'dummy': 'model'}
    save_model(model, 'model.pkl')
    loaded_model = load_model('model.pkl')
    logger.info(f"Model loaded: {loaded_model}")

    # Send email (example, replace with actual credentials)
    # send_email('Test Subject', 'Test Body', 'to@example.com', 'from@example.com',
    #            'smtp.example.com', 465, 'username', 'password')
