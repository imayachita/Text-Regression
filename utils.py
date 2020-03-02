import pandas as pd
import numpy as np
import seaborn as sns
import re
from matplotlib import pyplot as plt
from nltk import word_tokenize
import time
import os

def clean_specialLetters(cell):
    """
    Cleaning out special characters and non-unicode characters.

    Args:
        cell (str): input string
    Returns:
        clean (str): cleaned string
    """
    removed = re.sub('[^A-Za-z0-9]+', ' ', cell)
    clean = removed.encode("ascii", errors="ignore").decode()
    return clean


def load_stopwords(stopwords_file):
    """
    Load stopwords file

    Args:
        stopwords_file (str): .txt file that contains all stop words
    Returns:
        stopwords (list): stop words in a list
    """
    f = open(stopwords_file,'r',encoding='utf-8')
    stopwords = f.read().split('\n')
    f.close()
    return stopwords


def remove_stopwords(text,stopwords_file):
    """
    Removing stopwords from the text

    Args:
        text (list): input string in list
        stopwords_file (str): .txt file that contains all stop words
    Returns:
        filtered_words (list): cleaned string in list
    """

    stopwords = load_stopwords(stopwords_file)
    filtered_words = []
    for sentence in text:
        tokenized = word_tokenize(sentence)
        cleaned = [word for word in tokenized if word not in stopwords]
        cleaned = ' '.join(word for word in cleaned)
        filtered_words.append(cleaned)

    return filtered_words


def remove_numbers(cell):
    """
    Cleaning out numbers.

    Args:
        cell (str): input string
    Returns:
        cell (str): cleaned string
    """
    cell = re.sub('[0-9]+', '', cell)
    return cell


def clean_data(texts,stopwords_file=None):
    """
    Clean the text data by removing stopwords, numbers, and special characters

    Args:
        texts (list): input string in list
        stopwords_file (str): .txt file that contains all stop words
    Returns:
        cleaned_data (list): cleaned string in list
    """
    cleaned_data=[]
    for text in texts:
        pro_text = text.casefold()
        pro_text = clean_specialLetters(pro_text)
        pro_text = remove_numbers(pro_text)
        cleaned_data.append(pro_text)

    if stopwords_file is not None:
        cleaned_data = remove_stopwords(cleaned_data,stopwords_file)
    return cleaned_data


def plot_barplot(df,col_name,dir):
    """
    plot class distribution
    Args:
        df (dataframe): dataframe to plot
        col_name (col_name): column name of dataframe to plot
        dir: directory to save the plot
    """
    plt.figure()
    plt.title('Plot class distribution. X-Axis represents the class.')
    sns.barplot(x=df[col_name].value_counts().index,y=df[col_name].value_counts())
    plt.xticks([])
    plt.savefig(os.path.join(dir,"barplot_"+col_name+".png"))



def read_embedding(embedding_model):
    '''
    Read word embedding file
    Convert it into dictionary with words as the keys and the corresponding vectors as the values
    Args:
        embedding_model: word embedding model in txt format
    Returns:
        embeddings_dict (dict): dictionary of words and its vectors
    '''
    print('Reading word embedding file..')
    embeddings_dict = {}
    fEmbeddings = open(embedding_model)
    for i,line in enumerate(fEmbeddings):
        if i == 0:
            continue
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
    embedding_dim = len(vector)
    print('Done reading word embedding file.')
    return embeddings_dict, embedding_dim
    

def pad_sentences_with_vectors(sents, embeddings_dict, max_len, dim):
    '''
    Zero padding sentences with length shorter than max_len,
    truncat sentences with length exceeding max_len

    Args:
        sents (list): list of sentences or strings
        embeddings_dict (dict): word embedding dictionary
        max_len (int): maximum number of words in a sentence or string
        dim (int): word embedding dimension
    Returns:
        padded (list): list of np.array with shape=[max_len, dim]
    '''

    start = time.time()
    print('Start padding..')
    padded = np.zeros((len(sents), max_len, dim), dtype=np.float32)
    for i in range(len(sents)):
        sent = sents[i]
        vecs = np.zeros((max_len, dim), dtype=np.float32)
        for j in range(min(len(sent), max_len)):
            try:
                vecs[j] = embeddings_dict[sent[j]]
            except:
                vecs[j]=0

        padded[i] = vecs
    end = time.time()
    print('Done padding')
    return padded


def create_embedding_features(
                                mode,
                                train,
                                test,
                                input_col,
                                target_col,
                                embeddings_dict,
                                embedding_dim,
                                stopwords_file=None,
                                max_len=100,
                                limit=None
                                ):
    '''
    Create features using embedding. Output is numpy array with shape [num_examples,maxlen,embedding_dim]
    Args:
        mode (str): mode whether it is "training", "eval", "prediction"
        train (dataframe): train data in dataframe format
        test (dataframe): test data in dataframe format
        input_col: name of feature column in the dataframe
        target_col: name of label column in the dataframe
        embeddings_dict: word embedding dictionary with words as the keys and the corresponding vectors as values
        embedding_dim: word embedding dimension
        stopwords_file: stopwords list in txt file
        max_len: maximum length of sentence. sentences longer than this will be truncated
        limit: maximum number of examples for each class
    Returns:
        X: train data with shape=[num_examples,max_len,embedding_dim]
        Y: train label with shape=[num_examples,]
        test_X: validation data with shape=[num_examples,max_len,embedding_dim]
        test_Y: validation label with shape=[num_examples,]
    '''
    X = clean_data(train[input_col],stopwords_file)

    if mode!="prediction": #mode training or evaluation
        Y = train[target_col].tolist()
        if limit != None and mode=="training": # Limit the number of each example. Training mode only
            X, Y, = limit_data(X, Y, limit=limit)
        Y = np.array(Y)
    else: #training or evaluation
        Y = None

    X = pad_sentences_with_vectors(X,embeddings_dict,max_len,embedding_dim)

    if mode=="training": #Training mode
        test_X = clean_data(test[input_col],stopwords_file)
        test_X = pad_sentences_with_vectors(test_X, embeddings_dict,max_len,embedding_dim)
        test_Y = np.array(test[target_col])
    else: #prediction or evaluation mode
        test_X = None
        test_Y = None

    return X,Y,test_X,test_Y


def limit_data(data, labels_vec, limit):
    '''
    Limit data for training to handle imbalanced data.
    Randomly sample training data.
    Maximum examples of each class = limit
    Args:
        data (list): list of input data
        labels_vec (list): list of labels for each corresponding data
        limit (int): max number of examples for each class
    Returns:
        shuffled_data (list): shuffled data with limited examples for each class
        shuffled_labels (list): corresponding labels
    '''
    rand = np.random.permutation(len(data)) #create list of random data index
    shuffled_data = []
    shuffled_labels = []
    label_counter = dict((l, 0) for l in labels_vec)

    for i in rand:
        label = labels_vec[i]
        if label_counter[label]<limit:
            shuffled_data.append(data[i])
            shuffled_labels.append(label)
            label_counter[label]+=1

    return shuffled_data, shuffled_labels
