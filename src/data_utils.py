import os
from typing import Tuple

import gdown
import pandas as pd

from src import config


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download from GDrive all the needed datasets for the project.

    Returns:
        train : pd.DataFrame
            Training dataset

        test : pd.DataFrame
            Test dataset
    """

    # Download application_test_aai.csv
    if not os.path.exists(config.DATASET_TEST):
        gdown.download(config.DATASET_TEST_URL, config.DATASET_TEST, quiet=False)

    # Download application_train_aai.csv
    if not os.path.exists(config.DATASET_TRAIN):
        gdown.download(config.DATASET_TRAIN_URL, config.DATASET_TRAIN, quiet=False)

    train = pd.read_csv(config.DATASET_TRAIN)
    test = pd.read_csv(config.DATASET_TEST)

    return train, test


def split_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Args:
        train : pd.DataFrame
            Training dataset.
        test : pd.DataFrame
            Test dataset.

    Returns:
        X_train : pd.Series
            List reviews for train

        y_train : pd.Series
            List labels for train

        X_test : pd.Series
            List reviews for test

        y_test : pd.Series
            List labels for test
    """
    # Define the feature column and the target column
    feature_column = "review"    # Column containing the reviews (input features)
    target_column = "positive"   # Column indicating if the review is positive (1) or negative (0)

    # Split the training dataset into features and targets
    X_train = train[feature_column]   # Reviews for training
    y_train = train[target_column]    # Positive labels for training

    # Split the test dataset into features and targets
    X_test = test[feature_column]     # Reviews for testing
    y_test = test[target_column]      # Positive labels for testing

    return X_train, y_train, X_test, y_test

