import os
import pandas as pd

from src.data.Trajectory import Trajectory


def load(data_directory):
    """
    Reads data from CSV files in given directory to Trajectory objects

    :param data_directory: contains CSV files with datetime, latitude, longitude
    :return: list of Trajectory objects
    """
    csv_files = [
        os.path.join(data_directory, file)
        for file in os.listdir(data_directory)
        if file.endswith("csv")
    ]
    return [
        Trajectory(pd.read_csv(csv))
        for csv in csv_files
    ]


def apply_filters(trajectories, filters):
    """

    :param trajectories:
    :param filters:
    :return:
    """
    # TODO: implement mobility.apply_filters()
    if filters is None or len(filters) == 0:
        return trajectories
    else:
        return trajectories


def resample(trajectories, settings):
    """

    :param trajectories:
    :param settings:
    :return:
    """
    # TODO: implement mobility.resample()
    return trajectories
