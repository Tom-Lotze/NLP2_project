from math import isclose
import numpy as np
import os


class Generator(object):
    '''
    Generate samples using the terminals and operators that are given, and the ruleset for the probabilities
    '''
    def __init__(self, terminals, operators):
        self.terminals = terminals
        self.operators = operators
        self.ruleset = self.create_ruleset()
        self.training_data = []


    def generate(self, nr_samples, testing=False):
        """
        generate self.nr_samples samples and return them in a list
        """
        list_of_samples = set()
        if testing:
            self.removed_duplicates = 0

        while len(list_of_samples) < nr_samples:

            if len(list_of_samples) % 1000 == 0:
                print(f"{len(list_of_samples)} generated")
            sample = 'S'
            cont = True

            while cont:
                sample, cont = self.expand_sample(sample)


            if self.validate_seq(sample, testing):
                list_of_samples.add(sample)

        # save the training data to remove duplicates from test data
        if not testing:
            self.training_data = list_of_samples


        return list_of_samples


    def create_ruleset(self):
        """
        Create a ruleset with the probabilities for every rule, and validate it
        """
        nr_letters = len(self.terminals)
        ruleset = dict()

        ruleset['S'] = {'Fu S': 7/12, 'Fb Y S': 1/12, 'X': 3/12, 'S + S': 1/12}
        # very high probability for + for testing, above one is more balanced
        # ruleset['S'] = {'Fu S': 1/12, 'Fb Y S': 1/12, 'X': 5/12, 'S + S': 5/12}

        ruleset['Fu'] = {'F1': 1/9, 'F2': 1/9, 'F3': 1/9,
                         'B1': 1/9, 'B2': 1/9, 'B3': 1/9,
                         'R' : 1/9, '@' : 1/9, '#' : 1/9}
        ruleset['Fb'] = {'SHIFT ' : 1.0}
        ruleset['Y'] = {letter : 1 / nr_letters for letter in self.terminals}
        ruleset['X'] = {'X X': 3/8, 'Y': 5/8}
        ruleset['+'] = {'+': 1.0}

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


    def validate_seq(self, seq, testing):
        """
        Add constraints for the sequences that are generated, e.g. length, specific combinations etc.
        """
        if len(seq) > 40:
            return False

        if testing:
            if seq in self.training_data:
                self.removed_duplicates += 1
                return False

        return True




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
                    return string

                else: # unary operator
                    string = operator_fn(string)


        return string



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






if __name__ == '__main__':

    nr_train_samples = 50000
    nr_test_samples = 2000

    ops_set = {'F1', 'F2', 'F3', 'B1', 'B2', 'B3', 'R', '@', '#', 'SHIFT', '+'}

    # terminals = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    terminals = ['a','b','c','d','e','f']


    GEN = Generator(terminals, ops_set)
    PAR = Parser(terminals)

    data = {}

    os.makedirs("data", exist_ok=True)

    # generate training data
    data["src-train"] = GEN.generate(nr_samples=nr_train_samples)
    data["tgt-train"] = [PAR.parse_seq(seq) for seq in data["src-train"]]

    print("Training data generated...")

    data["src-test"] = GEN.generate(nr_samples=nr_test_samples, testing=True)
    data["tgt-test"] = [PAR.parse_seq(seq) for seq in data["src-test"]]

    print(f"Test data generated, {GEN.removed_duplicates} duplicates removed...")


    for name, dataset in data.items():
        with open(f'data/{name}.txt', 'w') as f:
            for x in dataset:
                f.write(f'{x}\n')

    print("Data is saved to ./data")


