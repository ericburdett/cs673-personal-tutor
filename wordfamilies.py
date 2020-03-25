import os

class EnglishWordFamilies:
    """Word families for English.

    Currenty based on the txt file in data/wordfamilies.txt."""
    def __init__(self):
        self._families = self._parse_txt()
        self._rev_families = self._reverse_families(self._families)

    def _parse_txt(self):
        """Parse the text file, returning a dict of word families."""
        fileloc = os.path.join('data', 'wordfamilies.txt')
        families = {}
        with open(fileloc) as infile:
            word = None
            family = set()
            for line in infile:
                if line.startswith('#'):
                    continue
                if not line.startswith('\t'):
                    families[word] = family
                    word = line.strip()
                    family = set()
                else:
                    family.add(line.strip())
            families[word] = family
        return families

    @staticmethod
    def _reverse_families(d):
        ret_d = dict()
        for k, s in d.items():
            for w in s:
                ret_d[w] = k
        return ret_d

    def get_base(self, word):
        """Returns the base word for the given word.

        If no word family is found, the given word is returned.
        """
        return self._rev_families.get(word, word)

    def get_family(self, word):
        """Returns the family of the given word.

        If no word family is found, the set {word} is returned.
        """
        return self._families.get(self.get_base(word), {word})


if __name__ == '__main__':
    pass
