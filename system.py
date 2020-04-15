from collections import Counter, namedtuple
import os
import re
import textwrap

try:
    from termcolor import colored
    COLORED = True
except:
    print('install termcolor to use colored output')
    def colored(s, *args, **kwargs):
        return s
    COLORED = False

Token = namedtuple('Token', 'tok loc')

class ReaderSystem:
    def __init__(self):
        self._strip = '.?!'
        self.width = max(min(30, self._get_win_width()-3), 70)

        with open('data/contractions.txt') as infile:
            _contractions = infile.read().split()

        self._words_re = re.compile(
            '('+'|'.join(_contractions + [r'\w+']) + ')',
            flags=re.IGNORECASE)

    def _get_tokens(self, s):
        # return sentence.strip(self._strip).split()
        #matches = list(re.finditer(r'\w+', s, flags=re.IGNORECASE))
        matches = list(self._words_re.finditer(s))
        tokens = []
        for match in matches:
            loc = (match.start(), match.end())
            tok = s[loc[0]: loc[1]]
            tokens.append(Token(tok, loc))
        return tokens

    def _get_tok_to_ind(self, tokens):
        """Get dict of token to indices, lowercasing all tokens.
        """
        tok_to_ind = {}
        i = 0
        for t in tokens:
            tok = t.tok.lower()
            if tok not in tok_to_ind:
                tok_to_ind[tok] = i
                i += 1
        return tok_to_ind

    def _get_formatters(self, tokens):
        """Get a list of formatters for the given tokens such that
        the numbers over the top will line up with the words."""
        lengths = []
        tok_to_ind = self._get_tok_to_ind(tokens)
        for i, (tok, loc) in enumerate(tokens):
            length = max(len(str(tok_to_ind[tok.lower()])), len(tok))
            lengths.append(length)
        return ['{:<' + str(ln) + '}' for ln in lengths]

    def _get_win_width(self):
        """Gets the width of the current terminal window."""
        rows, cols = (int(x) for x in os.popen('stty size', 'r').read().split())
        return cols

    def _wrap(self, text):
        return textwrap.wrap(text, width=self.width)

    def _get_text(self, tokens,
        lines_between=1, line_fill='-',
        tok_colors=None):
        """Gets the text for the numbers and words.

        tokens: the list of tokens
        lines_bewteen: number of lines between each line of text
        line_fill: what to fill the line with
        tok_colors: {lower case token: text color} dict
        """

        line = line_fill*self.width+'\n'
        tok_to_ind = self._get_tok_to_ind(tokens)

        if tok_colors is None:
            tok_colors = {}
        formatters = self._get_formatters(tokens)

        # Below we just concat a bunch of strings
        # Definitely not the optimal way to handle it,
        #   but since our strings are not very big,
        #   it doesn't really matter

        pos = 0  # column count

        # all the text, the text line for the numbers,
        # and the text for the sentence
        all_text = nums_text = sent_text = ''
        for i, (f, (tok, loc)) in enumerate(zip(formatters, tokens)):
            lower = tok.lower()

            # Move to next line
            if pos + len(tok) > self.width:
                all_text += nums_text + '\n'
                all_text += sent_text + '\n'
                all_text += line*lines_between
                nums_text = sent_text = ''
                pos = 0

            # Add the coloring
            if lower in tok_colors:
                nums_text += colored(f.format(tok_to_ind[lower]),
                                     tok_colors[lower])
            else:
                nums_text += f.format(tok_to_ind[lower])
            sent_text += f.format(tok)
            pos += len(tok)

            # Fill in the missing pieces that aren't contained
            #   in tokens, such as symbols and spcaes
            a = loc[1]
            b = tokens[i+1].loc[0] if i < (len(tokens)-1) else None
            missing = self.sentence[a:b]
            nums_text += ' '*len(missing)
            pos += len(missing)
            sent_text += missing
        all_text += nums_text + '\n'
        all_text += sent_text + '\n'
        return all_text

    def read_sentence(self, sentence,
                      lines_between=1, line_fill='-'):
        """Read the sentence, getting user input.

        Wasn't sure what this should return, but probably a
        list of tokens that were unknown? Not sure about the
        casing, since the language model cares about that.

        Lower case tokens would be in tok_colors.
        """

        self.sentence = sentence

        tokens = self._get_tokens(sentence)
        tok_to_ind = self._get_tok_to_ind(tokens)
        ind_to_tok = {v: k for k, v in tok_to_ind.items()}
        tok_colors = {}
        while True:
            os.system('clear')
            print('\n'*3)
            text = self._get_text(
                tokens, lines_between=lines_between,
                line_fill=line_fill, tok_colors=tok_colors)
            print(text)
            print('Type the index of words you don\'t know. q to quit\n',
                  '  ')
            inp = input().strip()
            if inp == 'q':
                break
            if inp.isdigit() and int(inp) in ind_to_tok:
                tok_colors[ind_to_tok[int(inp)]] = 'red'
            else:
                print('Unknown input or bad index:', inp)
                input('Press any key to continue')

            # Returning tokens in a format that the UserKnowledge class expects
            unknown_tokens = list(tok_colors)
            known_tokens = list(set(tokens) - set(unknown_tokens))
            all_tokens = np.concatenate((unknown_tokens, known_tokens))
            token_knowledge = np.concatenate((np.zeros(len(unknown_tokens)), np.ones(len(known_tokens))))

        return all_tokens, token_knowledge.astype(bool)



if __name__ == '__main__':
    system = ReaderSystem()


    sentence = ('Getting this to correctly apply the spacing and the coloring '
                'took much much longer than it should have. It\'s also a '
                'little tricky when you have random characters like !@#$#$%@# '
                'in the middle.')

    # tokens = system._get_tokens(sentence)
    # tok_to_ind = system._get_tok_to_ind(tokens)
    # formatter = system._get_formatter(tokens)
    # t = formatter.format(*[tok_to_ind[tok.lower()] for tok, _ in tokens])
    # s = formatter.format(*[tok for tok, _ in tokens])
    # print(t)
    # print(s)

    unknown_tokens = system.read_sentence(sentence)

