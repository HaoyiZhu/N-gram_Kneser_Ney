#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------
# Copyright (c) Haoyi Zhu. All rights reserved.
# Authored by Haoyi Zhu (hyizhu1108@gmail.com)
# -----------------------------------------------------

"""
NLP Big Homework 1: N-gram Language Model & Discounting Algorithm.
"""

import os
import time
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
# from tqdm import tqdm

train_set_path = './train_set.txt'
dev_set_path = './dev_set.txt'
test_set_path = './test_set.txt'

class NGramKneserNeyDiscountingModel():
    def __init__(self,
        train_set_path=train_set_path,
        dev_set_path=dev_set_path,
        test_set_path=test_set_path,
        max_n_gram_order=3,
        unk_threshold=1,
        delta=0.84,
        ):
        self._train_set_path = train_set_path
        self._dev_set_path = dev_set_path
        self._test_set_path = test_set_path
        self._max_n_gram_order = max_n_gram_order
        self._unk_threshold = unk_threshold
        self._delta = delta

    def _load_data(self, filename):
        with open(filename, 'r') as f:
            data = f.readlines()[0].split()

        return data

    def _counting_and_preprocess(self):
        print('Counting and preprocessing...')
        self.frequencies = [[]] + [{} for _ in range(self._max_n_gram_order)]
        self.prefixes = [[], []] + [{} for _ in range(self._max_n_gram_order - 1)]
        
        # Counting for N=1
        for word in self.train_word_list:
            if word in self.frequencies[1]:
                self.frequencies[1][word] += 1
            else:
                self.frequencies[1][word] = 1

        # Filter those words with low frequency and set them to <UNK>
        UNK_set = set()
        UNK_num = 0
        for key in self.frequencies[1]:
            if self.frequencies[1][key] <= self._unk_threshold:
                UNK_set.add(key)
                UNK_num += 1
        for key in UNK_set:
            self.frequencies[1].pop(key)

        self.frequencies[1]['<UNK>'] = UNK_num

        # Counting for N>1
        for N in range(2, self._max_n_gram_order + 1):
            ngram_list = self._generate_ngrams(self.train_word_list, N)
            for gram in ngram_list:
                _gram = [gram[i] if gram[i] not in UNK_set else '<UNK>' \
                         for i in range(len(gram))]
                key = tuple(_gram)
                if key in self.frequencies[N]:
                    self.frequencies[N][key] += 1
                else:
                    self.frequencies[N][key] = 1

                    prefix = tuple(_gram[:-1])
                    if prefix not in self.prefixes[N]:
                        self.prefixes[N][prefix] = 1
                    else:
                        self.prefixes[N][prefix] += 1

        return

    def _generate_ngrams(self, orig_word_list, n):
        word_list = copy.deepcopy(orig_word_list)
        for i in range(n - 1):
            word_list.insert(0, '<s>')
            word_list.append('</s')

        ngram_list = [word_list[i: i + n] \
                      for i in range(len(word_list) - (n - 1))]

        return ngram_list

    def _P_Kneser_Ney(self, word:str, prefixes:list):
        # For the highest order
        if prefixes == []:
            count = self.frequencies[1][word] if word in self.frequencies[1] else 0
            return (count / len(self.train_word_list))

        ngram = prefixes + [word]
            
        # Computing the numerator
        N = len(ngram)
        key = tuple(ngram)
        
        numerator_A = max(self.frequencies[N][key] - self._delta, 0) \
                          if key in self.frequencies[N] else 0

        # Computing the Denominator
        key = tuple(prefixes) if (len(key) == 1) else key[0]
        
        denominator = self.frequencies[N - 1][key] if key in self.frequencies[N - 1] else 1

        numerator_B = self.prefixes[N][tuple(prefixes)] if tuple(prefixes) in self.prefixes[N] else 0

        if numerator_B == 0:
            return (numerator_A / denominator)
        
        return (numerator_A / denominator) + self._delta * (numerator_B / denominator) * self.__P_Kneser_Ney(word, prefixes[1:])
  
    def __P_Kneser_Ney(self, word:str, prefixes:list):
        # For lower order
        if prefixes == []:
            continuation_count = self.prefixes[2][tuple([word])] if tuple([word]) in self.prefixes[2] else 0
            return (continuation_count / len(self.frequencies[2]))
        
        ngram = prefixes + [word]
            
        # Computing the numerator
        N = len(ngram)
        key = tuple(ngram)
        
        numerator_A = 0
        prefix = tuple(ngram[1:])

        numerator_A = self.prefixes[N][prefix] if prefix in self.prefixes[N] else 0

        # Computing the Denominator
        key = tuple(prefixes) if (len(key) == 1) else key[0]
        
        denominator = len(self.frequencies[N])
        
        numerator_B = self.prefixes[N][tuple(prefixes)] if tuple(prefixes) in self.prefixes[N] else 0

        if numerator_B == 0:
            return (numerator_A / denominator)
        
        return (numerator_A / denominator) + self._delta * (numerator_B / denominator) * self.__P_Kneser_Ney(word, prefixes[1:])

    def calculate_log10_probability_and_perplexity(self, sentence, gram_order=None):
        print('Start calculating probability (log10) and perplexity...')

        if isinstance(sentence, str):
            word_list = sentence.split()
        elif isinstance(sentence, list):
            word_list = sentence
        else:
            raise Exception(f'The input sentence should be in format of list ot str.')

        if gram_order is None:
            gram_order = self._max_n_gram_order

        n_grams_generated = self._generate_ngrams(word_list, gram_order)

        sentence_probability = 0
        perplexity = 0
        
        for i, gram in enumerate(n_grams_generated):
            _gram = [gram[i] if gram[i] in self.frequencies[1] else '<UNK>' \
                 for i in range(len(gram))]

            word = _gram[len(_gram) - 1]
            prefixes = _gram[0: len(_gram) - 1]
            if not isinstance(prefixes, list):
                prefixes = [prefixes]
            probability = self._P_Kneser_Ney(word, prefixes)

            if probability == 0:
                continue
            perplexity -= np.log2(probability)
            sentence_probability += np.log10(probability)
        
        perplexity = np.power(2, perplexity / i)
        
        return {'log10 probability' : sentence_probability, 'perplexity' : perplexity}

    def calculate_perplexity(self, sentence, gram_order=None):
        print('Start calculating perplexity...')
        
        if isinstance(sentence, str):
            word_list = sentence.split()
        elif isinstance(sentence, list):
            word_list = sentence
        else:
            raise Exception(f'The input sentence should be in format of list ot str.')

        if gram_order is None:
            gram_order = self._max_n_gram_order

        n_grams_generated = self._generate_ngrams(word_list, gram_order)

        perplexity = 0

        p_list = {}
        
        for i, gram in enumerate(n_grams_generated):
            _gram = [gram[i] if gram[i] in self.frequencies[1] else '<UNK>' \
                 for i in range(len(gram))]
            
            word = _gram[len(_gram) - 1]
            prefixes = _gram[0: len(_gram) - 1]
            if not isinstance(prefixes, list):
                prefixes = [prefixes]

            if tuple(_gram) not in p_list:
                probability = self._P_Kneser_Ney(word, prefixes)
                p_list[tuple(_gram)] = probability
            else:
                probability = p_list[tuple(_gram)]

            if probability == 0:
                continue
            perplexity -= np.log2(probability)
        
        perplexity = np.power(2, perplexity / i)
        
        return perplexity

    def calculate_log10_probability(self, sentence, gram_order=None):
        print('Start calculating probability (log10)...')
        
        if isinstance(sentence, str):
            word_list = sentence.split()
        elif isinstance(sentence, list):
            word_list = sentence
        else:
            raise Exception(f'The input sentence should be in format of list ot str.')

        if gram_order is None:
            gram_order = self._max_n_gram_order

        n_grams_generated = self._generate_ngrams(word_list, gram_order)

        sentence_probability = 0
        
        for i, gram in enumerate(n_grams_generated):
            _gram = [gram[i] if gram[i] in self.frequencies[1] else '<UNK>' \
                 for i in range(len(gram))]

            word = _gram[len(_gram) - 1]
            prefixes = _gram[0: len(_gram) - 1]
            if not isinstance(prefixes, list):
                prefixes = [prefixes]
            probability = self._P_Kneser_Ney(word, prefixes)

            if probability == 0:
                continue
            sentence_probability += np.log10(probability)
                
        return sentence_probability

    def calculate_probability(self, sentence, gram_order=None):
        print('Start calculating probability...')
        
        if isinstance(sentence, str):
            word_list = sentence.split()
        elif isinstance(sentence, list):
            word_list = sentence
        else:
            raise Exception(f'The input sentence should be in format of list ot str.')

        if gram_order is None:
            gram_order = self._max_n_gram_order

        n_grams_generated = self._generate_ngrams(word_list, gram_order)

        sentence_probability = 0
        
        for i, gram in enumerate(n_grams_generated):
            _gram = [gram[i] if gram[i] in self.frequencies[1] else '<UNK>' \
                 for i in range(len(gram))]

            word = _gram[len(_gram) - 1]
            prefixes = _gram[0: len(_gram) - 1]
            if not isinstance(prefixes, list):
                prefixes = [prefixes]
            probability = self._P_Kneser_Ney(word, prefixes)

            if probability == 0:
                continue
            sentence_probability += np.log10(probability)
                
        return 10 ** sentence_probability

    def plot_freq_rank(self, filename1='top10000.png', filename2='greater10000.png'):
        word_id = []
        frequency = []
        rank = 1
        for key, value in sorted(self.frequencies[1].items(), key=lambda kv: kv[1], reverse=True):
            word_id.append(rank)
            frequency.append(value)
            rank += 1

        num = 10000
        x = word_id[:num]
        y = frequency[:num]

        plt.figure(figsize = (12, 8))
        plt.title("Frequency v.s. Word Rank (Top 10000)", fontsize = 24)
        plt.xlabel("Rank", fontsize = 24)
        plt.ylabel('Frequency', fontsize = 24)
        plt.tick_params(labelsize=20)
        plt.plot(x, y, color='darkblue')
        plt.savefig(filename1)

        x = word_id[num:]
        y = frequency[num:]

        plt.figure(figsize = (12, 8))
        plt.title("Frequency v.s. Word Rank (> 10000)", fontsize = 24)
        plt.xlabel("Rank", fontsize = 24)
        plt.ylabel('Frequency', fontsize = 24)
        plt.tick_params(labelsize=20)
        plt.plot(x, y, color='darkblue')
        plt.savefig(filename2)

        return


    def train(self):
        print('############### Training ###############')
        time1 = time.time()

        self.train_word_list = self._load_data(self._train_set_path)
        self._counting_and_preprocess()

        print(f'Training is finished. Total training time is {time.time() - time1 :.4f} seconds.\n')

        return

    def dev(self, delta_range=[0.7,0.9], delta_step=0.02, unk_threshold_range=[1,3], order=None):
        # To choose the best hyperparameters on dev set
        print('Start adjusting hyperparameters on dev set...')

        if order is None:
            order = self._max_n_gram_order
        assert order <= self._max_n_gram_order, f'N-gram oder should be less than {self._max_n_gram_order}!'

        dev_word_list = self._load_data(self._dev_set_path)
        delta_candidates = [delta_range[0] + i * delta_step \
            for i in range(int((delta_range[1] - delta_range[0]) / delta_step) + 1)]
        unk_threshold_candidates = [unk_threshold_range[0] + i \
            for i in range(int(unk_threshold_range[1] - unk_threshold_range[0] + 1))]

        best_delta = delta_candidates[0]
        best_unk_threshold = unk_threshold_candidates[0]
        best_ppl = 1e100

        time1 = time.time()

        for unk_threshold in unk_threshold_candidates:
            self._unk_threshold = unk_threshold
            for delta in delta_candidates:
                self._delta = delta
                ppl_dev = self.calculate_perplexity(dev_word_list, order)
                if ppl_dev < best_ppl:
                    best_delta = delta
                    best_unk_threshold = unk_threshold
                    best_ppl = ppl_dev

        self._unk_threshold = best_unk_threshold
        self._delta = best_delta

        print('=========================================')
        print(f'Best delta: {best_delta}')
        print(f'Best UNK threshold: {best_unk_threshold}')
        print(f'Best DEV PPL: {best_ppl}')
        print(f'Total time: {time.time() - time1:.4f} s')
        print('=========================================\n')

        return

    def test(self, order=None):
        print('############### Testing ###############')
        if order is None:
            order = self._max_n_gram_order
        assert order <= self._max_n_gram_order, f'N-gram order should be less than {self._max_n_gram_order}!'

        print(f'Test n-gram oder is {order}.')
        time1 = time.time()

        test_word_list = self._load_data(self._test_set_path)
        ppl = self.calculate_perplexity(test_word_list, order)

        print(f'PPL on test set is {ppl}. Total testing time is {time.time() - time1:.4f} seconds.\n')

def save_model(m, filename='language_model.pkl'):
    with open(filename, 'wb') as f:
        f.write(pickle.dumps(m))
    print(f'Model saved in {filename}')

    return

def load_model(filename):
    with open(filename, 'rb') as f:
        m  = pickle.loads(f.read())

    return m

def main():
    # m = NGramKneserNeyDiscountingModel(max_n_gram_order=4)
    # m.train()
    # save_model(m)

    # m = load_model('language_model.pkl')

    # m.dev(order=2)
    # save_model(m, 'bi_gram_lm.pkl')
    # m.test(order=2)

    # m.dev(order=3)
    # save_model(m, 'tri_gram_lm.pkl')
    # m.test(order=3)

    # m.dev(order=4)
    # save_model(m, 'quad_gram_lm.pkl')
    # m.test(order=4)

    m = NGramKneserNeyDiscountingModel()
    m.train()
    m.test()
    save_model(m)


if __name__ == '__main__':
    main()