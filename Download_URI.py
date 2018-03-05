import urllib.request as urllib2
import requests, io, os
from io import StringIO
import numpy as np 
import tarfile, zipfile, gzip

def unzip_from_UCI(UCI_url, dest=''):
    """ Downloads and unpacks datasets from UCI in zip format """
    response = requests.get(UCI_url)
    compressed_file = io.BytesIO(response.content)
    z = zipfile.ZipFile(compressed_file)
    print ('Extracting in %s' % os.getcwd()+'\\'+dest)
    for name in z.namelist():
        if '.csv' in name:
            print('\tunzipping %s' % name)
            z.extract(name, path=os.getcwd()+'\\'+dest)


def gzip_from_UCI(UCI_url, dest=''):
    """ Downloads and unpacks datasets from UCI in gzip format """
    response = urllib2.urlopen(UCI_url)
    compressed_file = io.BytesIO(response.read())
    decompressed_file = gzip.GzipFile(fileobj=compressed_file)
    filename = UCI_url.split('/')[-1][:-3]
    with open(os.getcwd()+'\\'+filename, 'wb') as outfile:
        outfile.write(decompressed_file.read())
    print('File %s decompressed' % filename)

def targzip_from_UCI(UCI_url, dest='.'):
    """ Downloads and unpacks datasets from UCI in tar.gz format """ 
    response = urllib2.urlopen(UCI_url)
    compressed_file = StringIO(response.read())
    tar = tarfile.open(mode="r:gz", fileobj = compressed_file)
    tar.extractall(path=dest)
    datasets = tar.getnames()
    for dataset in datasets:
        size = os.path.getsize(dest+'\\'+dataset)
        print('File %s is %i bytes' % (dataset,size))
    tar.close()

def load_matrix(UCI_url):
    """ Downloads datasets from UCI in matrix form """ 
    return np.loadtxt(urllib2.urlopen(UCI_url))