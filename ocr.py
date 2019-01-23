#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#

# 1), HMM formulation
# In this case, we use the brown corpus to represent the statists of English languages
# For Initial Prob(IP), we filter valid chars and calculate the proportion
# For Transition Prob(TP), we iteratively pick the two nearby chars to count the proportion,
# --> In order to avoid the prob being too small, we use log method, and convert multiplication between two probs to the addition
# For Emission Pro(EP), we use a customized calc_similarity_score function, which compares two letter matrix in element wise method
# --> There are four cases, please see my implementation for detail
# --> Before we use this method, we actually a simple method which just counts the same pixels in two matrices, but the result is really poor
# We also wrote a debug helper, set OCR_DEBUG=1 (type export OCR_DEBUG=1 in your shell), and you can see the ground truth & accuracy very easily

# 2), How my program works
# My program first load the train and test images, convert them to pixel matrices,
# calculate Initial probabilities, Transition probabilities, Emission probabilities, which will be discussed later,
# then we use simple(char by char) and Viterbi algorithm to do the OCR respectively,
# finally, report the result
# We also wrote a debug helper, set OCR_DEBUG=1 (type export OCR_DEBUG=1 in your shell), and you can see the ground truth & accuracy very easily


# 3), Problems found and how to solve
# During the implementation, I first use a simple method to calculate the emission prob, n_mathed_pixels / (n_total_pixels).
# which works great on simple, but very poor on Viterbi, which boggles me very long time,
# I got a similar situation just like you and had just the same doubt,
# then I tried to use different formula when the pixel is True(*) or False(''), and it worked!
# to be more specific, there would be four different cases when comparing the training matrix(m1) and ground truth matrix(m2).
# * the pixel in m1 is True and the pixels in m2 is True
# * m1 is True, m2 is False
# * m1 is False, m2 is True
# * m1 is False, m2 is False
# I give them different score/use different formula and the result seems good, I am able to achieve 95%+ accuracy on Viterbi.
# I also applied the log method to avoid the number being underflow/too small.

# 4), Other questions ask in the Problem
# None

# Here we go


import numpy as np
from PIL import Image, ImageDraw, ImageFont
from string import ascii_uppercase, digits, ascii_lowercase
from collections import Counter, defaultdict
import sys
import math
import os


DEBUG = bool(os.environ.get('OCR_DEBUG'))
CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def calc_transmission_prob():
    """
        this function calculate transmission probability between two letters,
        the return variable is a defaultdict
        you can use ret[(char1, char2)] to get the transmission prob
    """
    counter = Counter()
    prob = defaultdict(lambda: 1.0e-7)

    valid_chars = set(TRAIN_LETTERS)
    text = open(train_txt_fname, 'r').read().strip()
    text = [c for c in text.replace('\n', ' ') if c in valid_chars]

    for c1, c2 in zip(text, text[1:]):
        counter[(c1, c2)] += 1

    total = len(text)
    for k, v in counter.items():
        prob[k] = float(v) / total
    return prob


def calc_emission_prob(test_matrices, train):
    res = []
    for matrix in test_matrices:
        res.append([
            calc_similarity_score(train[ch], matrix) for ch in TRAIN_LETTERS
        ])
    return res


def calc_inititial_prob():
    prob = defaultdict(lambda: 1.0e-7)

    valid_chars = set(TRAIN_LETTERS)
    text = open(train_txt_fname, 'r').read().strip()
    text = [c for c in text.replace('\n', ' ') if c in valid_chars]
    counter = Counter(text)
    total = len(text)

    for c in valid_chars:
        prob[c] = float(counter[c]) / total
    return prob


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size

    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        m = np.zeros((CHARACTER_WIDTH, CHARACTER_HEIGHT), dtype='bool')
        for x in range(x_beg, x_beg + CHARACTER_WIDTH):
            for y in range(0, CHARACTER_HEIGHT):
                m[x % CHARACTER_WIDTH, y] = True if px[x, y] < 1 else False
        result.append(m)
    return result


def load_training_letters(fname):
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(len(TRAIN_LETTERS))}


def do_simple_ocr(test_matrices, train):
    res = []
    candidates = [(i, x) for i, x in enumerate(TRAIN_LETTERS)]
    for index in range(len(test_matrices)):
        letter = sorted(
            candidates, key=lambda x: EP[index][x[0]], reverse=True)[0][1]
        res.append(letter)
    return ''.join(res)


def do_viterbi_ocr(observations, train):
    res = []
    candidates = list(TRAIN_LETTERS)

    first_prob = [np.log2(IP[ch] or 1.0e-7) + EP[0][i] for i, ch in enumerate(candidates)]
    probabilities = [first_prob]
    n_states = len(candidates)

    for i in range(1, len(observations)):
        previous_letter_prob = probabilities[-1]

        current_prob = np.zeros(n_states)
        for j in range(n_states):
            current_prob[j] = max([
                1.0 + np.e ** (previous_letter_prob[k]) + np.log2(TP[(candidates[k], candidates[j])]) + EP[i][j]
                for k in range(n_states)
            ])
        probabilities.append(current_prob)

    for p in probabilities:
        max_likelihood_index = np.argmax(p, axis=0)
        res.append(candidates[max_likelihood_index])
    return ''.join(res)


def logify(c):
    if isinstance(c, list):
        return [logify(i) for i in c]
    if c == 0:
        return 0
    return np.log(c)


def normalize(c):
    total = sum(c)
    return [i / total for i in c]


def calc_similarity_score(p1, p2):
    score = 0.0
    for i in range(p1.shape[0]):
        for j in range(p1.shape[1]):
            x, y = p1[i, j], p2[i, j]
            if x != y:
                if x == True:
                    score += np.log2(0.3)
                else:
                    score += np.log(0.01)
            else:
                if x == True:
                    score += np.log2(0.99)
                else:
                    score += np.log2(0.5)
    return score


def debug_helper(test_img_fname, your_result):
    """ This function read real letters from the folder
        and then test whether our simple/viterbi is operating properly
    """
    if not DEBUG:
        return

    fname = 'test-0-0.png test-1-0.png test-12-0.png test-14-0.png test-16-0.png test-18-0.png test-2-0.png test-4-0.png test-6-0.png test-8-0.png test-10-0.png test-11-0.png test-13-0.png test-15-0.png test-17-0.png test-19-0.png test-3-0.png test-5-0.png test-7-0.png test-9-0.png'
    fname_list = fname.split(' ')
    if test_img_fname not in fname_list:
        return
    import re
    index = int(re.findall(r'\d+', test_img_fname)[0])
    file = open('test-strings.txt', 'r')
    ground_truth = file.read().strip().split('\n')[index]

    accuracy = float(len([i for i in range(len(your_result))
                          if ground_truth[i] == your_result[i]])) / len(your_result)
    print(ground_truth)
    print('Accuracy: %', accuracy * 100)


if __name__ == '__main__':

    train_img_fname, train_txt_fname, test_img_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    train_letters = load_training_letters(train_img_fname)
    test_letters = load_letters(test_img_fname)
    TP = calc_transmission_prob()
    IP = calc_inititial_prob()
    EP = calc_emission_prob(test_letters, train_letters)

    simple_ocr_res = do_simple_ocr(test_letters, train_letters)
    viterbi_ocr_res = do_viterbi_ocr(simple_ocr_res, train_letters)
    final_res = viterbi_ocr_res
    print('Simple: ', simple_ocr_res)
    print('Viterbi: ', viterbi_ocr_res)
    print('Final answer:\n', final_res)
    debug_helper(test_img_fname, final_res)
