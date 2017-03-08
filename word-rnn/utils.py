# -*- coding: utf-8 -*-
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess_by_jokes(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches_by_jokes()
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        # print vocabulary
        return [vocabulary, vocabulary_inv]

    def preprocess_by_jokes(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        #data = self.clean_str(data)
        x_text = data.split()
        self.vocab, self.words = self.build_vocab(x_text)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        lines = data.split('\n')
        words_in_lines = []
        max_len = 0
        for line in lines:
            words = line.split()
            max_len = max(max_len, len(words))
            words_in_lines.append(words)
        print 'max length of jokes is ' + str(max_len)
        # print words_in_lines

        # convert to number and do padding
        uniform_len_lists = []
        for words in words_in_lines:
            num_list = list(map(self.vocab.get, words))
            num_zeros_to_add = max_len - len(num_list)
            for i in range(num_zeros_to_add):
                num_list.append(0)
            uniform_len_lists.append(num_list)
        # print uniform_len_lists

        self.tensor = np.array(uniform_len_lists)
        # print self.tensor
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        #data = self.clean_str(data)
        x_text = data.split()
        # print 'x_text'
        # print x_text
        self.vocab, self.words = self.build_vocab(x_text)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.tensor = np.array(list(map(self.vocab.get, x_text)))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def load_nouns(self):
        print 'loading nouns...'
        nouns_file = os.path.join(self.data_dir, "inputNN.txt")
        with open(nouns_file, "r") as f:
            string = f.read()
        f.close()
        nouns_lists = string.split('\n')
        nouns_matrix = []
        for line in nouns_lists:
            nouns = line.split()
            # nouns_in_num = list(map(self.vocab.get, nouns))
            nouns_matrix.append(nouns)
        print nouns_matrix


    def create_batches_by_jokes(self):
        """
        python train.py --data_dir data/joke --batch_size 2 --num_epochs 1 --seq_length 26
        seq_length must be equal to max joke length (max_len)
        """
        # self.batch_size = 1
        self.num_batches = self.tensor.shape[0] / self.batch_size
        # print self.num_batches
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        # print "***"
        # print self.tensor.reshape(1, -1)[0]
        # print self.tensor
        self.tensor = self.tensor.reshape(1, -1)[0]
        self.tensor = self.tensor[:self.seq_length * self.num_batches * self.batch_size]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        # list of arrays
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        # print self.load_nouns()
        # print 'self.x_batches:'
        # print self.x_batches
        # print 'self.y_batches:'
        # print self.y_batches

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # round to the multiple of self.batch_size * self.seq_length
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        # print "***"
        # print xdata
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        # list of arrays
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        # print 'self.x_batches:'
        # print self.x_batches
        # print 'self.y_batches:'
        # print self.y_batches

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        # print 'next_batch'
        # print 'x: '
        # print x
        # print 'y: '
        # print y
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
