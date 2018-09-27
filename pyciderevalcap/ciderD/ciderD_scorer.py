#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy
import json
import math
import os

from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np


PUNCTUATIONS = ["''", "'", "``", "`", "(", ")", "{", "}", "[", "]", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]


def term_frequency(sentence, ngrams=4):
    """Given a sentence, calculates term frequency of tuples.

    Parameters
    ----------
    sentence : str
        Sentence whose term frequency has to be calculated.
    ngrams : int
        Number of n-grams for which term frequency is calculated.

    Returns
    -------
    dict
        {tuple : int} key-value pairs representing term frequency.
    """
    sentence = sentence.lower().strip()
    for punc in PUNCTUATIONS:
        sentence = sentence.replace(punc, "")
    words = TreebankWordTokenizer().tokenize(sentence)
    counts = {}
    for i in range(ngrams):
        for j in range(len(words) - i):
            ngram = tuple(words[j:(j + i + 1)])
            if ngram in counts:
                counts[ngram] += 1
            else:
                counts[ngram] = 1
    return counts


def cook_refs(refs, n=4):
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [term_frequency(ref, n) for ref in refs]


def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return term_frequency(test, n)


class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0, df_mode="coco-val-df"):
        """Singular instance."""
        self.n = n
        self.sigma = sigma
        self.df_mode = df_mode
        self.ctest = []
        self.crefs = []
        self.cook_append(test, refs)
        self.ref_len = None
        self.document_frequency = None

    def cook_append(self, test, refs):
        """Called by constructor and __iadd__ to avoid creating new instances."""
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if isinstance(other, tuple):
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def _compute_document_frequency(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        document_frequency = {}
        if self.df_mode == "corpus":
            for refs in self.crefs:
                # refs, k ref captions of one image
                for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                    document_frequency[ngram] += 1
            assert(len(self.ctest) >= max(document_frequency.values()))
        elif self.df_mode == "coco-val-df":
            document_frequency_temp = json.load(open(os.path.join('data', 'coco_val_df.json')))
            # convert string to tuple
            for key in document_frequency_temp:
                document_frequency[eval(key)] = document_frequency_temp[key]
        else:
            raise ValueError(f"df_mode can be either corpus or coco-val-df, provided {self.df_mode}!")
        return document_frequency

    def compute_score(self):
        self.document_frequency = self._compute_document_frequency()
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [{} for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(self.document_frequency.get(ngram, 1.0))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.  the norm will be used for
                # computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    val[n] += vec_hyp[n].get(ngram, 0) * vec_ref[n].get(ngram, 0)

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        # compute log reference length
        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))
        elif self.df_mode == "coco-val-df":
            # if coco option selected, use length of coco-val set
            self.ref_len = np.log(float(40504))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return np.mean(np.array(scores)), np.array(scores)
