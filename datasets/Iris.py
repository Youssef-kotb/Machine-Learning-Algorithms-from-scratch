import os
import urllib.request
import pandas as pd
import shutil


class Iris:
    def __init__(self):

        # Link to download the dataset from
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

        # defining directories
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.download_path = f"{BASE_DIR}/downloaded_datasets/Iris"
        self.file_path = f"{self.download_path}/Iris.data"

        # checking if the dataset is already downloaded
        if os.path.isdir(self.download_path):
            self.downloaded = True
        else:
            self.downloaded = False



        #info
        self.features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
        ]

        self.target = [
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica"
        ]            
            
    def load_data(self):
        
        if not self.downloaded:
            os.makedirs(self.download_path)

            urllib.request.urlretrieve(self.url, self.file_path)
            self.downloaded = True
        
        features, target = read_dataset(self.file_path)
        
        return features, target


    def redownload_data(self):

        shutil.rmtree(self.download_path)

        os.makedirs(self.download_path)

        urllib.request.urlretrieve(self.url, self.file_path)

        print ("Redownloaded the dataset successfully")


def read_dataset(path):

    columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]


    df = pd.read_csv(path, header = None, names = columns)

    features = df.iloc[:,:-1]
    target = df.iloc[:,-1]

    return features, target