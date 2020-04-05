from math import isclose
import numpy as np



class Generator(object):
    '''
    Generate samples using the terminals and operators that are given, and the ruleset for the probabilities
    '''
    def __init__(self, terminals, operators, nr_samples=100):
        self.terminals = terminals
        self.operators = operators
        self.ruleset = self.create_ruleset()
        self.nr_samples = nr_samples


    def generate(self):
        """
        generate self.nr_samples samples and return them in a list
        """
        list_of_samples = []

        while len(list_of_samples) < self.nr_samples:
            sample = 'S'
            cont = True

            while cont:
                sample, cont = self.expand_sample(sample)

            # modify to remove spaces in the character string
            out = self.remove_spaces(sample)

            if self.validate_seq(out):
                list_of_samples.append(out)

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


    def validate_seq(self, seq):
        """
        Add constraints for the sequences that are generated, e.g. length, specific combinations etc.
        """
        if len(seq) > 20:
            return False

        return True

    def remove_spaces(self, seq):
        """
        Rearrange the tokens in the sequence so that all tokens are seperated by spaces, except for terminals (letters), which will not be separated.
        """
        out = ''
        prev_token = None

        for token in seq.split():
            if token not in self.terminals or prev_token == 'SHIFT':
                if token == '+':
                    out += ' ' + token + ' '
                else:
                    out += token + ' '
            else:
                out += token
            prev_token = token

        return out.strip()



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
        splitted = raw_seq.split()

        if len(splitted) == 1:
            # print(f'len1 triggered on {splitted}')
            return splitted[0]

        seq = splitted[-1]
        operations = splitted[:-1]


        for i, token in enumerate(operations[::-1]):
            if token in self.operators.keys():
                # define operator function
                operator_fn = self.operators[token]

                if operator_fn == self.shift:
                    seq = operator_fn(seq, shift_factor)

                elif operator_fn == self.concatenate:
                    prepend_seq = self.remove_spaces(' '.join(operations[:-i-1]))
                    seq = operator_fn(self.parse_seq(prepend_seq), self.parse_seq(seq))
                    return seq

                else: # unary operator
                    seq = operator_fn(seq)

            else: # shift is the next open
                shift_factor = self.terminals.index(token) + 1

        return seq

    def remove_spaces(self, seq):
        """
        Rearrange the tokens in the sequence so that all tokens are seperated by spaces, except for terminals (letters), which will not be separated.
        """
        out = ''
        prev_token = None

        for token in seq.split():
            if token not in self.terminals or prev_token == 'SHIFT':
                if token == '+':
                    out += ' ' + token + ' '
                else:
                    out += token + ' '
            else:
                out += token
            prev_token = token

        return out.strip()


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

    ops_set = {'F1', 'F2', 'F3', 'B1', 'B2', 'B3', 'R', '@', '#', 'SHIFT', '+'}

    # terminals = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    terminals = ['a','b','c','d','e','f']


    GEN = Generator(terminals, ops_set, nr_samples=100)
    PAR = Parser(terminals)

    samples = GEN.generate()
    labels = [PAR.parse_seq(seq) for seq in samples]

    with open('generated_data.txt', 'w') as f:
        for x, y in zip(samples, labels):
            f.write(f'{x}.{y}\n')



