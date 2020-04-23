from math import isclose
import numpy as np
import os
from random import shuffle


class Generator(object):
    '''
    Generate samples using the terminals and operators that are given, and the ruleset for the probabilities
    '''
    def __init__(self, terminals, operators, max_length, training):
        self.terminals = terminals
        self.operators = operators
        self.max_length = max_length
        self.training = training
        self.training_duplicate = 0

        self.ruleset = self.create_ruleset()


    def generate(self, nr_samples):
        """
        generate nr_samples samples and return them in a list
        """
        sample_set = set()
        printed = set()

        while len(sample_set) < nr_samples:

            # print progress
            curr_length = len(sample_set)
            if curr_length % 50 == 0 and curr_length not in printed:
                print(f"{curr_length} generated")
                printed.add(curr_length)

            sample = 'S'
            cont = True

            while cont:
                sample, cont = self.expand_sample(sample)

            if self.validate_seq(sample):
                sample_set.add(sample)

        # convert to list and shuffle
        ALL_DATA = list(sample_set)
        shuffle(ALL_DATA)
        self.ALL_DATA = ALL_DATA

        print("Duplicates filtered out:", self.training_duplicate)

        return ALL_DATA


    def create_ruleset(self):
        """
        Create a ruleset with the probabilities for every rule, and validate it
        """
        nr_letters = len(self.terminals)
        ruleset = dict()

        ruleset['S'] = {'Fu S': 7/12, "Fb Y S" : 1/12, 'X': 4/12}


        ruleset['Fu'] = {'F1': 1/9, 'F2': 1/9, 'F3': 1/9,
                         'B1': 1/9, 'B2': 1/9, 'B3': 1/9,
                         'R' : 1/9, '@' : 1/9, '#' : 1/9}
        ruleset['Fb'] = {'SHIFT ' : 1.0}

        ruleset['Y'] = {letter : 1 / nr_letters for letter in self.terminals}
        ruleset['X'] = {'X X': 3/8, 'Y': 5/8}

        # validate the ruleset
        for dic in ruleset.values():
            assert(isclose(sum(dic.values()), 1))
        return ruleset


    def expand_sample(self, seq):
        """
        Iteratively expand a sample by applying rules to the non-terminal symbols
        """
        out = []
        cont = False

        for token in seq.split()[::-1]:
            if token in self.terminals or token in self.operators:
                out.append(token)
                continue

            cont = True
            rule = self.choose_rule(token)
            out.append(rule)

        out = ' '.join(out[::-1])

        return out, cont


    def choose_rule(self, token):
        """
        Choose one of the rules for the token according to the probabilities
        """
        options = self.ruleset[token]
        chosen_rule = np.random.choice(list(options.keys()), p=list(options.values()))

        return chosen_rule


    def validate_seq(self, seq):
        """
        Add constraints for the sequences that are generated, e.g. length, specific combinations etc.
        """
        if seq in self.training:
            self.training_duplicate += 1
            return False

        # make sure there are exactly 10 operators in the sequence
        if self.nr_operators(seq) != 9:
            return False

        return True


    def nr_operators(self, seq):
        return len([i for i in seq.split() if not i.islower()])



class Parser(object):
    """
    Parser class,
    """
    def __init__(self, terminals):
        self.terminals = terminals
        self.operators = self.create_ops_dict()

        self.nr_terminals = len(terminals)


    def create_ops_dict(self):
        return {'F1': self.forward_1, 'F2': self.forward_2,
                'F3': self.forward_3, 'B1': self.backward_1,
                'B2': self.backward_2, 'B3': self.backward_3,
                'R': self.reverse, '@': self.last_to_front,
                '#': self.first_to_end, 'SHIFT': self.shift,
                '+': self.concatenate}


    def parse_seq(self, raw_seq):
        """
        Parse a sequence accordint to the operator functions and return the outcome as a string
        """
        splitted = raw_seq.split()
        # print(f"raw_seq: {raw_seq}")

        if len(splitted) == 1:
            return splitted[0]

        # seperate the string of characters and the operations to apply on it
        string = ""

        for i, token in enumerate(splitted[::-1]):
            if token in self.terminals:
                string = token + string
                prev_token = token
                continue
            elif token in self.operators.keys():

                # define operator function
                operator_fn = self.operators[token]

                if operator_fn == self.shift:
                    # set the shift factor from the previous token
                    shift_factor = self.terminals.index(prev_token) + 1
                    # remove the shift factor from the character string
                    string = string[1:]
                    # appy the shift operation
                    string = operator_fn(string, shift_factor)

                elif operator_fn == self.concatenate:
                    # separate the first argument
                    prepend_seq = ' '.join(splitted[:-i-1])
                    # concatenate the parsed first and second argument and return
                    string = operator_fn(self.parse_seq(prepend_seq), self.parse_seq(string))
                    return self.format(string)

                else: # unary operator
                    string = operator_fn(string)


        return self.format(string)


    def format(self, string):
        return string.replace(" ", "").replace("", " ")[1:-1]


    def shift(self, string, x):
        ''' Shift the string with x steps'''
        return ''.join([self.terminals[(self.terminals.index(char) + x) % self.nr_terminals] for char in string])

    def forward_1(self, seq):
        return self.shift(seq, 1)

    def forward_2(self, seq):
        return self.shift(seq, 2)

    def forward_3(self, seq):
        return self.shift(seq, 3)

    def backward_1(self, seq):
        return self.shift(seq, -1)

    def backward_2(self, seq):
        return self.shift(seq, -2)

    def backward_3(self, seq):
        return self.shift(seq, -3)

    def reverse(self, seq):
        return seq[::-1]

    def last_to_front(self, seq):
        return seq[-1] + seq[:-1]

    def first_to_end(self, seq):
        return seq[1:] + seq[0]

    def concatenate(self, seq1, seq2):
        return seq1 + seq2


def split_between_ops(seq, ops):
    splitted = seq.split()

    out = []

    for i, el in enumerate(splitted):
        if el == "SHIFT":
            out.append(el + " " + splitted[i+1])
        elif el in ops:
            out.append(el)
        elif el.islower() and splitted[i-1] == "SHIFT":
            continue
        else:
            break

    terms = ""
    for term in splitted[i:]:
        terms += f"{term} "

    terms = terms[:-1]

    out.append(terms)

    return out




if __name__ == '__main__':
    # set variables for nr of samples and the train-test split
    nr_samples = 2000
    max_length = -1

    # define operator set
    ops_set = {'F1', 'F2', 'F3', 'B1', 'B2', 'B3', 'R', '@', '#', 'SHIFT'}

    terminals = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    nr_terminals = len(terminals)

    # make sure the new dataset does not contain any training instances
    with open("data/26term_-1max/train_src.txt") as f:
        training_data = [seq.strip() for seq in f.readlines()]


    # initiate generator and parser
    GEN = Generator(terminals, ops_set, max_length, training_data)
    PAR = Parser(terminals)

    # create necessary directories
    datadir = f"data/localism_sequential"
    os.makedirs("data", exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    data = {}

    # generate source data. NOTE: due to the strict requirements (10 ops per sequnce) this generation takes longer. Progress will be printed.
    data["complete"] = GEN.generate(nr_samples=nr_samples)
    data["TGT"] = [PAR.parse_seq(seq) for seq in data["complete"]]
    data["first"] = []
    data["second"] = []
    data["third"] = []
    data["fourth"] = []
    data["fifth"] = []
    data["sixth"] = []
    data["seventh"] = []
    data["eighth"] = []
    data["nineth"] = []
    data["tenth"] = []


    for seq in data["complete"]:
        [one, two, three, four, five, six, seven, eight, nine, ten] = split_between_ops(seq, ops_set)
        data["first"].append(one)
        data["second"].append(two)
        data["third"].append(three)
        data["fourth"].append(four)
        data["fifth"].append(five)
        data["sixth"].append(six)
        data["seventh"].append(seven)
        data["eighth"].append(eight)
        data["nineth"].append(nine)
        data["tenth"].append(ten)



    # save the generated data
    for name, dataset in data.items():
        with open(f'{datadir}/{name}.txt', 'w') as f:
            for x in dataset:
                f.write(f'{x}\n')





