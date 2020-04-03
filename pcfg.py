# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-04-03 16:25
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-04-03 20:11


from math import isclose
import numpy as np


# terminals = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
terminals = ['a','b','c','d','e','f']

def shift(seq, x):
    ''' Shift the sequence with x steps'''
    return ''.join([terminals[(terminals.index(char) + x) % len(terminals)] for char in seq])

def forward_1(seq):
    return shift(seq, 1)

def forward_2(seq):
    return shift(seq, 2)

def forward_3(seq):
    return shift(seq, 3)

def backward_1(seq):
    return shift(seq, -1)

def backward_2(seq):
    return shift(seq, -2)

def backward_3(seq):
    return shift(seq, -2)

def reverse(seq):
    return seq[::-1]

def last_to_front(seq):
    return seq[-1] + seq[:-1]

def first_to_end(seq):
    return seq[1:] + seq[0]



def create_ruleset(terminals):
    nr_letters = len(terminals)
    ruleset = dict()

    ruleset['S'] = {'Fu S': 6/12, 'Fb S Y': 1/12, 'X': 5/12}
    ruleset['Fu'] = {'F1': 1/6, 'F2': 1/6, 'F3': 1/6,
                     'B1': 1/6, 'B2': 1/6, 'B3': 1/6}
    ruleset['Fb'] = {'SHIFT' : 1.0}
    ruleset['Y'] = {letter : 1/nr_letters for letter in terminals}

    ruleset['X'] = {'X X': 1/4, 'Y': 3/4}

    # validate the ruleset
    for dic in ruleset.values():
        assert(isclose(sum(dic.values()), 1))

    return ruleset




def generator(ruleset, nr_samples, terminals, operators):
    for i in range(nr_samples):
        sample = "S"
        cont = True

        while cont:
            sample, cont = expand_sample(sample, ruleset, terminals, operators)

        print(sample)



def expand_sample(sample, ruleset, terminals, operators):
    out = []
    cont = False

    for token in sample.split()[::-1]:
        if token in terminals or token in operators.keys():
            out.append(token)
            continue

        cont = True

        rule = choose_rule(ruleset, token)
        out.append(rule)

    out = ' '.join(out[::-1])

    return out, cont


def choose_rule(ruleset, token):
    options = ruleset[token]
    chosen_rule = np.random.choice(list(options.keys()), p=list(options.values()))

    return chosen_rule





def parser(seq):
    splitted = seq.split()

    for token in splitted:
        if token in operators.keys():
            operator_fn = operators[token]





if __name__ == "__main__":
    ruleset = create_ruleset(terminals)

    operators = {'F1': forward_1, 'F2': forward_2, 'F3': forward_3,
             'B1': backward_1, 'B2': backward_2, 'B3': backward_3,
             "R": reverse, "@": last_to_front, "#": first_to_end,
             "SHIFT": shift}


    generator(ruleset, 40, terminals, operators)















