from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing
import torch

import pandas as pd
import numpy as np

from dataset_preprocess import dataset_preprocess
from embeddings import create_embeddings, load_obj

import matplotlib.pyplot as plt


class ResumeDataset(Dataset):
    def __init__(self, df, embedding_matrix, word_to_ind_f, vocabulary, unk_word='unk'):
        self.df = df.copy()

        # list of Tensors with different lengths;
        # list of sentences;

        # sentence is list of words;
        # word is vector (embedded)

        self.df['Resume'] = self.df['Resume'].apply(lambda resume: [sentence for sentence in resume if sentence])
        self.df['Resume'] = self.df['Resume'].apply(lambda resume:
                                                    [torch.Tensor([embedding_matrix[word_to_ind_f[a]]
                                                                   if a in vocabulary
                                                                   else embedding_matrix[word_to_ind_f[unk_word]]
                                                                   for a in sentence])
                                                     for sentence in resume])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        return:
            resume -
            label - is int number
        """
        row = self.df.loc[idx, ['Resume', 'Category_label']]

        length = [int(sentence.shape[0]) for sentence in row['Resume']]
        # padding with 0 to same sizes
        resume = torch.nn.utils.rnn.pad_sequence(row['Resume'], batch_first=True)

        return resume, length, int(row['Category_label'])


def create_dataset(path, embedding_path=None, plotting=False):
    dataframe = pd.read_csv(path)
    dataframe = dataset_preprocess(dataframe, plotting=plotting)

    if embedding_path is None:
        emb_matrix, word_to_ind, ind_to_word, ind_to_freq, vocabulary = create_embeddings(dataframe, run_test=False)
    else:
        emb_matrix, word_to_ind, ind_to_word, ind_to_freq, vocabulary = np.load(embedding_path), \
                                                                        load_obj('word_to_ind'), \
                                                                        load_obj('ind_to_word'), \
                                                                        load_obj('ind_to_freq'), \
                                                                        load_obj('vocabulary')

    # Label encoding for category
    le = preprocessing.LabelEncoder()
    le.fit(dataframe['Category'].values)
    print(le.classes_)
    dataframe['Category_label'] = dataframe['Category'].apply(lambda x: le.transform([x])[0])

    # train test split
    train_split = 1 - 0.15

    train_df = []
    test_df = []
    for category in dataframe['Category'].unique():
        category_df = dataframe[dataframe['Category'] == category]

        train_category_df = category_df[:int(len(category_df) * train_split)]
        test_category_df = category_df[int(len(category_df) * train_split):]

        train_df.append(train_category_df)
        test_df.append(test_category_df)

    train_df = pd.concat(train_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)

    if plotting:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        fig.autofmt_xdate(rotation=25)

        pd.value_counts(train_df['Category']).plot.bar(ax=ax[0], rot=25)
        ax[0].set_title('Train number of documents (cv)')

        pd.value_counts(test_df['Category']).plot.bar(ax=ax[1], rot=25)
        ax[1].set_title('Test number of documents (cv)')
        plt.show()

    train_all_dataset = ResumeDataset(train_df, emb_matrix, word_to_ind, vocabulary)
    test_all_dataset = ResumeDataset(test_df, emb_matrix, word_to_ind, vocabulary)

    train_dataloader = DataLoader(train_all_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_all_dataset, batch_size=1, shuffle=True)

    return emb_matrix, word_to_ind, ind_to_word, ind_to_freq, vocabulary, train_all_dataset, test_all_dataset, \
           train_dataloader, test_dataloader, dataframe

