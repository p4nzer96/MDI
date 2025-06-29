import random
from typing import Iterable

import pandas as pd
from pandas import DataFrame


def _get_available_pairs(data: pd.DataFrame) -> set:
    """ Returns all the subjects which have at least one morphed image

    :param data: The input dataframe containing the dataset index
    :type data: pd.DataFrame
    :return: A set of subjects which have at least one morphed image
    :rtype: set
    """

    subjects: set = _get_unique_subjects(data)

    subjects_wo_morphs: set = set()

    for s in list(subjects):
        condition = (data["ID1"] == s) & (data["ID2"].notna())
        if data.loc[condition].empty:
            subjects_wo_morphs.add(s)

    return subjects - subjects_wo_morphs


def _get_unique_subjects(data: pd.DataFrame) -> set:
    """ Returns all unique subjects in the dataset

    :param data: The input dataframe containing the dataset index
    :type data: pd.DataFrame
    :return: A set of unique subjects
    :rtype: set
    """
    return set(pd.unique(data["ID1"]))


def get_split(data: pd.DataFrame, subjects: Iterable) -> pd.DataFrame:
    """ Returns a subset of the dataset containing only the specified subjects
    
    :param data: The input dataframe containing the dataset index
    :type data: pd.DataFrame
    :param subjects: The subjects to include in the subset
    :type subjects: Iterable
    :return: A subset of the dataset containing only the specified subjects
    :rtype: pd.DataFrame
    """
    return data.loc[(data.ID1.isin(subjects)) & (data.ID2.isin(subjects) | data.ID2.isna())].reset_index(drop=True)


def get_train_val_test_split(data: pd.DataFrame, train_split=0.7, val_split=0.1, strict=False) -> tuple[
    DataFrame, DataFrame | None, DataFrame]:
    """ Splits the dataset into training, validation, and test sets based on the specified splits
    :param data: The input dataframe containing the dataset index
    :type data: pd.DataFrame
    :param train_split: The proportion of the dataset to include in the training set
    :type train_split: float
    :param val_split: The proportion of the dataset to include in the validation set
    :type val_split: float
    :param strict: If True, only subjects with morphed images will be considered for training
    :type strict: bool
    :return: A tuple containing the training, validation, and test sets
    :rtype: tuple[DataFrame, DataFrame | None, DataFrame]
    """

    subjects: set = _get_unique_subjects(data)
    if strict:
        available_training_subjects = _get_available_pairs(data)
    else:
        available_training_subjects = subjects

    train_subjects = random.sample(list(available_training_subjects), int(len(subjects) * train_split))
    if val_split != 0.0:
        validation_subjects = random.sample(list(train_subjects), int(len(subjects) * val_split))
        # Ensure validation subjects are not in training set
        train_subjects = list(set(train_subjects) - set(validation_subjects))
    else:
        validation_subjects = set()
    test_subjects = list(set(subjects) - set(train_subjects) - set(validation_subjects))

    train_data = get_split(data, subjects=train_subjects)
    val_data = get_split(data, subjects=validation_subjects) if val_split != 0.0 else None
    test_data = get_split(data, subjects=test_subjects)

    return train_data, val_data, test_data
