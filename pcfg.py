# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-04-03 16:25
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-04-03 16:41



terminals = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}


def shift(seq, x):
    """ Shift the sequence with x steps"""
    return ''.join(chr((ord(char) - 97 + x) % 26 + 97) for char in seq)

def reverse(seq):
    return seq[::-1]

def last_to_front(seq):
    return seq[-1] + seq[:-1]

def first_to_end(seq):
    return seq[1:] + seq[1]

