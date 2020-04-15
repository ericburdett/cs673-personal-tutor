import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import spacy
import pandas as pd

"""## Word Distribution"""
class WordDist():
  """Word Distribution read from a file stored in a pd.DataFrame.

  File expected to be a csv of the form word,count on each line.
  """
  def __init__(self, input_file='data/word_dist_full.csv'):
    self.df = pd.read_csv(input_file, header=None, names=['word', 'freq'])

  def getdf(self):
    return self.df

  def dict_normalized(self):
    """Returns a dict of {word: normalized freqency}."""
    copy = self.df.copy()
    copy['freq'] = copy['freq'] / copy['freq'].max()

    return copy.set_index('word').to_dict()['freq']

  def __getitem__(self, index):
    if isinstance(index, int):
      return self.df['word'][index], self.df['freq'][index]
    elif isinstance(index, str):
      # Returns tuples as above. There may be a better way
      return tuple((self.df.loc[self.df['word'] == index].values)[0])

    else:
        raise ValueError('Unknown index type')

  def __len__(self):
    return len(self.df)


"""## Language Model and Evaluation Classes"""

class LanguageModel():
  def __init__(self, mask=None, k=50):
    self.model = GPT2LMHeadModel.from_pretrained('distilgpt2').cuda()
    self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    self.k = k

    if mask == None:
      self.mask = torch.ones(len(self.tokenizer.get_vocab())).cuda()
    else:
      self.mask = torch.tensor(mask).cuda()

  def top_k_logits(self, logits):
    if self.k == 0:
        return logits
    values, _ = torch.topk(logits, self.k)
    min_values = values[-1]
    # Why the -1e10?
    return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)

  def tokenizer(self):
    return self.tokenizer

  def set_mask(self, mask):
    self.mask = torch.tensor(mask).cuda()

  def get_text(self, prompt, length):
    """Returns text generated from the prompt of the given length (words)."""
    generated = self.tokenizer.encode(prompt)
    context = torch.tensor([generated]).cuda()
    past = None
    for i in range(length):
      output, past = self.model(context, past=past)

      logits = output[..., -1, :].squeeze()
      logits = logits * self.mask # Apply the mask to the logits

      topk_logits = self.top_k_logits(logits)
      topk_log_probs = F.softmax(topk_logits, dim=-1)
      token = torch.multinomial(topk_log_probs, num_samples=1)

      generated += [token.item()]
      context = token.unsqueeze(0)

    sequence = self.tokenizer.decode(generated)
    return sequence

  def get_sentence(self, prompt, length):
    """Returns a sentence from the prompt of at most length."""
    sequence = self.get_text(prompt, length)
    end_index = len(prompt.split('. '))
    return ".".join(sequence.split('.')[0:end_index]) + '.'

  def get_sentences(self, prompt, sentence_length, num_sentences):
    """Returns sentences from get_sentence."""
    sentences = []

    for i in range(num_sentences):
      sentence = self.get_sentence(prompt, sentence_length)
      sentences.append(sentence)

    return sentences



class Evaluator():
  def __init__(self, topic, spacy_model):
    self.space_model = spacy_model
    self.topic_doc = spacy_model(topic)

  def get_keywords(self, sentence_doc):
    # Find Nouns and Adjectives
    keywords = []
    for token in sentence_doc:
      pos = token.pos
      if pos in [92, 96]: # NOUN, PNOUN, ADJ , 84
         keywords.append(token)

    return keywords

  def get_random_pairs(self, arr, size):
    """Returns `size` random pairs of items from `arr`."""
    # FIXME Probably a better way to do this
    pairs = []

    try:
      for i in range(size):
        pair = np.random.choice(arr, size=2, replace=False)
        pairs.append(pair)
    except:
      return None

    return pairs

  def generate_scores_array(self, size):
    """Returns points array for ranking purposes."""
    # FIXME This is a little odd, but I like what it's going for
    # See the difference between size==9 and size==10.
    step = size // 10 + 1
    return np.array([i for i in range(11) for _ in range(step)])[::-1][:size]

  def get_score(self, sentences, eval_func, negate=False):
    """Gets the ranked scores from the sentences using the eval function
    and generate_scores_array.
    """
    lengths = [eval_func(sentence) for sentence in sentences]
    sort_indices = np.argsort(np.negative(lengths)) if negate else np.argsort(lengths)
    scores = self.generate_scores_array(len(sentences))
    np.put(scores, sort_indices, scores)
    return scores

  def sentence_length_score(self, sentences):
    """Ranked scores sentences based on len(sentence), shorter better."""
    return self.get_score(sentences, len, negate=False)

  def topic_score(self, sentences):
    return self.get_score(sentences, self.single_topic_score, negate=True)

  def related_score(self, sentences):
    return self.get_score(sentences, self.single_related_score, negate=True)

  def score_sentences(self, sentences):
    """Scores sentences based on several tests. Returns average score."""
    nlp_sentences = [self.spacy_model(sentence) for sentence in sentences]

    scores = []
    scores.append(self.sentence_length_score(sentences))
    scores.append(self.topic_score(nlp_sentences))
    scores.append(self.related_score(nlp_sentences))
    # Append scores here for more tests

    return np.mean(scores, axis=0)

  def single_topic_score(self, sentence_doc):
    """Topic sort function."""
    keywords = self.get_keywords(sentence_doc)
    if len(keywords) < 2:
      return 0

    similarities = []
    for keyword in keywords:
      if keyword.vector_norm:
        similarity = self.topic_doc.similarity(keyword)
      else:
        similarity = 0

      similarities.append(similarity)

    return np.mean(similarities)

  def single_related_score(self, sentence_doc):
    """Related sort function."""
    keywords = self.get_keywords(sentence_doc)

    # Sample Random Pairs
    pairs = self.get_random_pairs(keywords, 10) # 10 seems like a good number for now...
    if pairs == None or len(pairs) == 0:
      return 0

    if len(keywords) != len(set(keywords)): # If the list contains duplicates, give poor score
      return 0

    # Check Similarity
    similarities = []
    for pair in pairs:

      if pair[0].vector_norm and pair[1].vector_norm:
        similarity = pair[0].similarity(pair[1])
      else:
        similarity = 0

      if similarity >= 1: # Do not give a high similarity score if we are comparing a word with itself
        similarity = 0

      similarities.append(similarity)
      # print('Comparing {} with {}, score: {:.4f}'.format(pair[0], pair[1], similarity))

    # print(similarities)

    return np.mean(similarities)


