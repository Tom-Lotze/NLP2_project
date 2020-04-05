# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-04-03 16:25
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-04-04 14:21


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
    return shift(seq, -3)

def reverse(seq):
    return seq[::-1]

def last_to_front(seq):
    return seq[-1] + seq[:-1]

def first_to_end(seq):
    return seq[1:] + seq[0]

def concatenate(seq1, seq2):
    return seq1 + seq2



def create_ruleset(terminals):
    nr_letters = len(terminals)
    ruleset = dict()

    ruleset['S'] = {'Fu S': 7/12, 'Fb Y S': 1/12, 'X': 3/12, "S + S": 1/12}

    # very high probability for + for testing, above one is more balanced
    # ruleset['S'] = {'Fu S': 1/12, 'Fb Y S': 1/12, 'X': 5/12, "S + S": 5/12}

    ruleset['Fu'] = {'F1': 1/9, 'F2': 1/9, 'F3': 1/9,
                     'B1': 1/9, 'B2': 1/9, 'B3': 1/9,
                     "R" : 1/9, "@" : 1/9, "#" : 1/9}
    ruleset['Fb'] = {'SHIFT ' : 1.0}
    ruleset['Y'] = {letter : 1/nr_letters for letter in terminals}

    ruleset['X'] = {'X X': 3/8, 'Y': 5/8}
    ruleset["+"] = {"+": 1.0}

    # validate the ruleset
    for dic in ruleset.values():
        assert(isclose(sum(dic.values()), 1))

    return ruleset


def generator(ruleset, nr_samples, terminals, operators):
    list_of_samples = []

    while len(list_of_samples) < nr_samples:
        sample = "S"
        cont = True

        while cont:
            sample, cont = expand_sample(sample, ruleset, terminals, operators)

        # modify to remove spaces in the character string
        out = remove_spaces(sample, terminals)

        if validate_seq(out):
            list_of_samples.append(out)

    return list_of_samples


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


def validate_seq(seq):
    if len(seq) > 20:
        return False

    return True


def remove_spaces(seq, terminals):
    out = ""
    prev_token = None

    for token in seq.split():
        if token not in terminals or prev_token == "SHIFT":
            if token == "+":
                out += " " + token + " "
            else:
                out += token + " "
        else:
            out += token
        prev_token = token

    return out.strip()


def parser(string, operators, terminals):
    splitted = string.split()

    if len(splitted) == 1:
        # print(f"len1 triggered on {splitted}")
        return splitted[0]

    seq = splitted[-1]
    operations = splitted[:-1]


    for i, token in enumerate(operations[::-1]):
        if token in operators.keys():
            operator_fn = operators[token]
            if operator_fn == shift:
                seq = operator_fn(seq, shift_factor)
            elif operator_fn == concatenate:
                prepend_seq = remove_spaces(" ".join(operations[:-i-1]), terminals)
                seq = operator_fn(parser(prepend_seq, operators, terminals), parser(seq, operators, terminals))
                return seq
            else:
                seq = operator_fn(seq)
        else: # shift is the next open
            shift_factor = terminals.index(token) + 1

    return seq




if __name__ == "__main__":
    ruleset = create_ruleset(terminals)

    operators = {'F1': forward_1, 'F2': forward_2, 'F3': forward_3,
             'B1': backward_1, 'B2': backward_2, 'B3': backward_3,
             "R": reverse, "@": last_to_front, "#": first_to_end,
             "SHIFT": shift, "+": concatenate}


    samples = generator(ruleset, 100, terminals, operators)

    labels = [parser(seq, operators, terminals) for seq in samples]

    with open("generated_data.txt", "w") as f:
        for x, y in zip(samples, labels):
            f.write(f"{x}.{y}\n")

    # testing specific sentence
    # parser("aae + B2 e", operators, terminals)












