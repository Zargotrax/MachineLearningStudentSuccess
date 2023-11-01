import numpy as np
import pandas as pd


class Dataset(object):
    _instance = None
    _raw_data = None

    def __new__(cls):
        if cls._instance is None:
            print('Loading new dataset')
            cls._instance = super(Dataset, cls).__new__(cls)
            cls._raw_data = pd.read_csv("./dataset.csv")
            # TODO : setup the data properly and stuff
            # Put any initialization here.
        return cls._instance

    def get_data(self):
        # TODO : return the proper data
        data = self._raw_data
        target = self._raw_data["Target"].values
        return (data, target)
