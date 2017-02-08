"""
Helper function to download and processing Stanford Sentiment Treebank datasets
"""

import os
from py.utils.io_utils import download, unzip, get_path

sst_url = "http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"


def download_sst(path):
    """
    Download zip file from url and extract all the files under path
    :param path:
    :return:
    """
    print("downloading ")
    local_file = os.path.join(path, 'stanfordSentimentTreebank.zip')
    download(sst_url, local_file)
    unzip(local_file, path)




if __name__ == '__main__':
    download_sst(get_path('cached_data/'))
