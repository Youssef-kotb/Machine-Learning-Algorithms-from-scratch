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
        self.downloaded = os.path.exists(self.file_path)


       
            
    def load_data(self):
        
        if not self.downloaded:
            os.makedirs(self.download_path)
            
            try:
                urllib.request.urlretrieve(self.url, self.file_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")

            self.downloaded = True
        
        features, target = read_dataset(self.file_path)
        
        return features, target


    def redownload_data(self):

        shutil.rmtree(self.download_path)

        os.makedirs(self.download_path)

        urllib.request.urlretrieve(self.url, self.file_path)

        print ("Redownloaded the dataset successfully")

    def info(self, download=False):
        info = {
            "name": "Iris",
            "task": "Classification",
            "source": "UCI ML Repository",
            "downloaded": self.downloaded
        }

        if self.downloaded or download:
            if not self.downloaded:
                self.load_data()
            columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
            df = pd.read_csv(self.file_path, header=None, names=columns)
            info.update({
                "n_samples": len(df),
                "n_features": 4,
                "features": columns[:-1],
                "target": "species",
                "class_distribution": df["species"].value_counts().to_dict()
            })
        else:
            info["data_info"] = (
                "Dataset not downloaded. "
                "Call info(download=True) or load_data() to see data-dependent info."
            )

        return info

    
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