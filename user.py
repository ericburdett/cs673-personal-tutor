import numpy as np
from collections import Counter

"""## User Knowledge"""
class WordMemory:
  def __init__(self, rank, memory_size=10):
    self.rank = rank
    self.memory_size = memory_size
    self.memory = np.zeros(memory_size).tolist()

  def add(self, is_known):
    # For now, only allow bools to be added to memory
    assert isinstance(is_known, bool)

    self.memory.append(int(is_known))

    # We are only remembering the past 'memory_size' times we've seen this word
    if len(self.memory) > self.memory_size:
      self.memory.pop(0)

  def ratio(self):
    if len(self.memory) == 0:
      return -1 # indicates the word has never been seen by the user
    else:
      return np.mean(self.memory)

  # FIXME This is overwritten by the __init__ `self.rank = rank`
  # Not sure what it was originally for
  def rank(self):
    return self.rank


class UserKnowledge:
  def __init__(self, vocabulary_list, tokenizer, focus=500,
               lower_bound=.10, upper_bound=1.):
    # contains dictionary of word : WordMemoryObject
    self.ranking = dict(enumerate(vocabulary_list))
    self.knowledge = {word: WordMemory(rank=index)
                      for index, word in self.ranking.items()}

    self.tokenizer = tokenizer
    self.tokenizer_vocab = tokenizer.get_vocab()

    self.focus = focus
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def update(self, words, knowns):
    """Update words and whether or not user knew them."""
    for word, known in zip(words, knowns):
      if word in self.knowledge: # Check first if the word exists in the dictionary
        self.knowledge[word].add(bool(known))
      # else: ## TODO ##
      #   Possibly add it to the knowledge base and then give it more likelihood to be shown again...

  def compute_ratios(self): # Known/Total
    """Return dict of {word: ratio}."""
    return {word: memory.ratio() for word, memory in self.knowledge.items()}

  def _get_punc_indicies(self):
    punc_indices = []
    punc_indices.append(self.tokenizer_vocab[self.tokenizer.tokenize(".")[0]])
    # punc_indices.append(self.tokenizer_vocab[self.tokenizer.tokenize(",")[0]])
    punc_indices.append(self.tokenizer_vocab[self.tokenizer.tokenize("!")[0]])
    # punc_indices.append(self.tokenizer_vocab[self.tokenizer.tokenize(";")[0]])
    return punc_indices

  def get_known_words(self):
    known = []
    for word, knowledge in self.knowledge.items():
      if knowledge.ratio() > .9:
        known.append(word)
    return known

  def get_focus_indices(self):
    inds = []
    count = 0
    for ind, word in self.ranking.items():
      ratio = self.knowledge[word].ratio()
      if ratio != 1 and count < self.focus:
        count += 1
        inds.append(ind)
    return inds

  def get_focus_words(self):
    inds = self.get_focus_indices()
    return [self.ranking[i] for i in inds]

  def get_focus_tokens(self, lower=True, capitalize=True,
                       duplicates=False):
    words = self.get_focus_words()
    return self.get_tokens(words, lower=lower,
                           capitalize=capitalize,
                           duplicates=duplicates)

  def get_tokens(self, words, lower=True, capitalize=True,
                 duplicates=False):
    # TODO Should be always using the space?
    tokens = Counter()
    for word in words:
      if lower:
        updates = self.tokenizer.tokenize(' ' + str(word))
        tokens.update(updates)
        # if len(updates)>1:
        #   print(word, updates)
      if capitalize:
        updates = self.tokenizer.tokenize(' ' + str(word).capitalize())
        tokens.update(updates)
        # if len(updates)>1:
        #   print(word, updates)
    if duplicates:
      # return a list with an element for each occurance
      return [x for k, v in tokens.items() for x in [k]*v]
    return list(tokens)

  def compute_mask(self):
    mask = np.ones(len(self.tokenizer_vocab))
    mask[:] = self.upper_bound
    indices = []

    # Give higher probability to punctuation so we don't end up with
    # really long sentences
    punc_indices = self._get_punc_indicies()
    for mask_index in punc_indices:
      mask[mask_index] = self.lower_bound

    # Get tokens for the given word with a space pre-pended and
    # also with the word capitalized
    tokens = self.get_focus_tokens(lower=True, capitalize=True)

    mask_indices = [self.tokenizer_vocab[t] for t in tokens]
    mask[mask_indices] = self.lower_bound
    # for mask_index in mask_indices:
    #   if (mask_value != 1):
    #     indices.append(mask_index)
    #     mask[mask_index] = mask_value

      # If the user hasn't learned a given word, give it a higher
      # probability for the language model to produce it

      # If we still need to fill the focus, try the most frequent words first

      # As the user learns the easier words, the more difficult
      # words will become more probable
    return mask




  # def compute_mask(self):
  #   mask = np.ones(len(self.tokenizer_vocab))
  #   indices = []
  #   current_num = 0

  #   # Give higher probability to punctuation so we don't end up with
  #   # really long sentences
  #   punc_indices = self._get_punc_indicies()
  #   for mask_index in punc_indices:
  #     mask[mask_index] = self.lower_bound

  #   # Iterate through the rankings in order...
  #   for index, word in self.ranking.items():
  #     # if current_num <= self.focus:
  #       # print(word, ',', current_num)

  #     # Get tokens for the given word with a space pre-pended and
  #     # also with the word capitalized
  #     tokens = self.get_focus_tokens(lower=True, capitalize=True)


  #     mask_indices = []
  #     for token in tokens:
  #       mask_indices.append(self.tokenizer_vocab[token])

  #     # If the user hasn't learned a given word, give it a higher
  #     # probability for the language model to produce it

  #     # If we still need to fill the focus, try the most frequent words first

  #     # As the user learns the easier words, the more difficult
  #     # words will become more probable
  #     ratio = self.knowledge[word].ratio()
  #     if ratio != 1.0 and current_num <= self.focus:
  #       mask_value = self.lower_bound
  #       current_num += 1
  #     else:
  #       mask_value = self.upper_bound

  #     for mask_index in mask_indices:
  #       if (mask_value != 1):
  #         indices.append(mask_index)
  #         mask[mask_index] = mask_value

  #   return mask


