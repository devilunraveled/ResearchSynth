import pandas as pd
import os

class DataLoader:
    def __init__(self, datasetName='arxiv'):
        '''
        Inputs:
            datasetName: name of the dataset to load. Options: 'arxiv', 'pubmed'.
        '''
        self.datasetName = datasetName

    def getData(self, datasetPath, split='train'):
        '''
        Inputs:
            datasetPath: path to the dataset folder relative to the current directory.
            split: name of the split to load. Options: 'train', 'val', 'test'.
        Returns:
            df: pandas dataframe with the dataset.
        '''
        # read the data from the jsonl file into a pandas dataframe
        dataPath = os.path.join(datasetPath, self.datasetName + '/' + split + '.txt')
        df = pd.read_json(dataPath, lines=True)

        return df