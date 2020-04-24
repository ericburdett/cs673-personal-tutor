from collections import namedtuple, Counter
import itertools
import pickle
import os
import sys
import time

import numpy as np
import spacy

from language import WordDist, LanguageModel, Evaluator
from user import UserKnowledge
from system import ReaderSystem

import torch

RANDOM_SEED = 0
TOPK = 50
MAX_SENTENCE_LENGTH = 60
NUM_SENTENCES = 20
KNOWN = 0

# USER OPTIONS
FOCUS = 500
LOWER_BOUND = 0.9
MEMORY_SIZE = 10
FOLDER = 'outdata3'


for arg in sys.argv:
    if 'known' in arg:
        KNOWN = int(arg.split('=')[1])
    if 'lower' in arg:
        LOWER_BOUND = float(arg.split('=')[1])


    # def __init__(self, TOPK=50,
    #              MAX_SENTENCE_LENGTH=60,
    #              NUM_SENTENCES=25,
    #              KNOWN=0, FOCUS=500,
    #              LOWER_BOUND=0.9,):
class Options:
    ARGS = ('TOPK', 'MAX_SENTENCE_LENGTH',
            'NUM_SENTENCES', 'KNOWN',
            'FOCUS', 'LOWER_BOUND',
            'RANDOM_SEED')
    def __init__(self, **kwargs):
        for arg in self.ARGS:
            setattr(self, arg, kwargs[arg])

    def asdict(self):
        return {arg: getattr(self, arg) for arg in self.ARGS}

    def __str__(self):
        return (
            'Options('
            + ', '.join(f'{arg}={getattr(self, arg)}' 
                        for arg in self.ARGS)
            + ')')

    def copy(self):
        return self.__class__(**self.asdict())



OPTIONS = Options(**{arg: globals()[arg] for arg in Options.ARGS})
# ({
#     'TOPK': 50,
#     'MAX_SENTENCE_LENGTH': 60,
#     'NUM_SENTENCES': 25,
#     'KNOWN': 0,
#     'FOCUS': 500,
#     'LOWER_BOUND': 0.9,
#     'RANDOM_SEED': 0
#     })


# OPTIONS = namedtuple('OPTIONS',
#     'TOPK '
#     'MAX_SENTENCE_LENGTH '
#     'NUM_SENTENCES '
#     'KNOWN '
#     'FOCUS '
#     'LOWER_BOUND '
#     )


# for arg in Options.ARGS:
#     if 'RANDOM' in arg:
#         continue
#     print(arg, getattr(OPTIONS, arg))
print()
# print(f'TOPK: {O.TOPK}')
# print(f'MAX_SENTENCE_LENGTH: {O.MAX_SENTENCE_LENGTH}')
# print(f'NUM_SENTENCES: {O.NUM_SENTENCES}')
# print(f'KNOWN: {O.KNOWN}')
# print()

NEWS_POLITICS_PROMPTS = ['President Trump said',
                         'CNN News Report:']
NEWS_POLITICS_TRUNCATE = [False,
                          True]

SPORTS_PROMPTS = ['ESPN Breaking News: ']
SPORTS_TRUNCATE = [True]

CORONAVIRUS_PROMPTS = ['This virus ',
                       'President Trump released a statement on the corona virus, saying: ',
                       'The World Health Organization said this in a statement Tuesday: ',
                       'The pandemic has ']
CORONAVIRUS_TRUNCATE = [False, True, True, False]

def remove_prompt(sentence, prompt):
  new_sentence = sentence.split(prompt)
  if len(new_sentence) < 2:
    return ''
  else:
    return sentence.split(prompt)[1]

def get_topic(topic_num):
  if topic_num == 0: # News/Politics
    prompts = NEWS_POLITICS_PROMPTS
    prompts_truncate = NEWS_POLITICS_TRUNCATE
    topic = "News"
  elif topic_num == 1: # Sports
    prompts = SPORTS_PROMPTS
    prompts_truncate = SPORTS_TRUNCATE
    topic = "Sports"
  elif topic_num == 2:
    prompts = CORONAVIRUS_PROMPTS
    prompts_truncate = CORONAVIRUS_TRUNCATE
    topic = "Health"
  else:
    raise ValueError('bad topic somehow')
  return prompts, prompts_truncate, topic



def run_system(O, topic_num=None, accept_input=True, n_sentences=None, nlp=None):
  # Topic 0
  #Nyc has more than the national and private health insurance market today.
  print(O)
  torch.manual_seed(O.RANDOM_SEED)
  np.random.seed(O.RANDOM_SEED)

  if topic_num is None:
    print("Select a Topic:")
    print("[0]: News/Politics")
    print("[1]: Sports")
    print("[2]: Coronavirus")
    while topic_num not in [0, 1, 2]:
      try:
        topic_num = int(input("Topic Number: "))
      except ValueError:
        print('Topic should be a number')

  print('\nLoading Language Learning System...')

  reader_system = ReaderSystem()

  if nlp is None:
    nlp = spacy.load('en_core_web_md')

  lm = LanguageModel(k=O.TOPK)
  lm.set_mask(np.ones(len(lm.tokenizer.get_vocab())))

  word_dist = WordDist()


  user_knowledge = UserKnowledge(word_dist.dict_normalized(), lm.tokenizer,
                                 focus=O.FOCUS,
                                 lower_bound=O.LOWER_BOUND)

  sorted_words = sorted(
    ((k, v) for k, v in word_dist.dict_normalized().items()),
    key=lambda kv: kv[1], reverse=True)

  for i in range(O.KNOWN):
    user_knowledge.update([sorted_words[i][0]]*MEMORY_SIZE, [True]*MEMORY_SIZE)

  prompts, prompts_truncate, topic = get_topic(topic_num)

  evaluator = Evaluator(topic, nlp)

  best_sentences = []

  i_sentence = 0
  start = time.time()
  n_sentences = float('inf') if n_sentences is None else n_sentences
  while True:
    i_sentence += 1
    if i_sentence > n_sentences:
      break
    try:
      print('Generating Sentence...', f'{i_sentence}/{n_sentences}',
            round(time.time()-start, 1), 'sec')

      # Select a random prompt from above
      rand_index = np.random.randint(len(prompts))
      prompt = prompts[rand_index]
      should_truncate = prompts_truncate[rand_index]

      # Get/set the mask
      new_mask = user_knowledge.compute_mask()
      lm.set_mask(new_mask)

      # Get sentences from the language model
      sentences = lm.get_sentences(prompt, O.MAX_SENTENCE_LENGTH, O.NUM_SENTENCES)
      if should_truncate:
        sentences = [remove_prompt(sentence, prompt) for sentence in sentences]

      # Score the sentences based on evaluation criteria
      scores = evaluator.score_sentences(sentences)
      best_sentence = sentences[np.argmax(scores)]

      best_sentences.append({'text': best_sentence,
                             'i_sentence': i_sentence,
                           })

      # Show the sentence to the user
      words, word_knowledge, inputs = reader_system.read_sentence(
        best_sentence, return_input=True,
        accept_input=accept_input)
      focus_words = set(user_knowledge.get_focus_words())
      known_words = set(user_knowledge.get_known_words())
      best_sentences[-1] = {'text': best_sentence,
                            'words': words,
                            'in_focus': [w in focus_words for w in words],
                            'in_known': [w in known_words for w in words],
                            'knowledge': word_knowledge,
                            'inputs': inputs,
                            'i_sentence': i_sentence,
                            }

      # Update our user knowledge and recompute the language model mask
      if accept_input:
          user_knowledge.update(words, word_knowledge)
    except KeyboardInterrupt:
      break
    except Exception as e:
      print('Another error occurred')
      print(e)
      break
  data = {'sentences': best_sentences,
          'topic_num': topic_num,
          'topic': topic,
          'options': O.asdict(),
         }
  return data

def get_options_product(base_options, **kwargs):
    optionslist = []
    product = itertools.product(*[[(k, v) for v in kwargs[k]]
                                  for k in kwargs if kwargs[k]])
    for p in product:
        o = base_options.copy()
        for k, v in p:
            setattr(o, k, v)
        optionslist.append(o)
    return optionslist

def get_practice_vars(options=OPTIONS):
  # PROBABLY NOT BEST PRACTICE
  # print('Building Language model')
  lm = LanguageModel(k=options.TOPK)
  lm.set_mask(np.ones(len(lm.tokenizer.get_vocab())))

  print('Getting word distribution')
  word_dist = WordDist()


  uk = UserKnowledge(word_dist.dict_normalized(), lm.tokenizer)

  sorted_words = sorted(
    ((k, v) for k, v in word_dist.dict_normalized().items()),
    key=lambda kv: kv[1], reverse=True)
  for i in range(options.KNOWN):
    uk.update([sorted_words[i][0]]*MEMORY_SIZE, [True]*MEMORY_SIZE)

  import inspect
  print('exporting to global')
  (next(frm[0] for frm in inspect.stack()
        if 'In' in frm[0].f_locals)).f_locals.update(locals())

if __name__ == "__main__":
  data = run_system(OPTIONS.copy(), None, accept_input=True, n_sentences=None)
  sys.exit()

  n = 0
  for arg in sys.argv:
    if arg.startswith('n='):
      n = int(arg.split('=')[1])

  print('Loading language model')
  nlp = spacy.load('en_core_web_md')
  # nlp = None
  MAXN = 10
  if n == 0:
    optionslist = [OPTIONS.copy()]
    n_sentences = None
  elif n in range(1, MAXN+1) or range(50,100):
    n_sentences = 50
    kwargs = {
      'TOPK': [50, 100],
      'MAX_SENTENCE_LENGTH': [],
      'NUM_SENTENCES': [],
      'KNOWN': [100, 500, 2000, 5000],
      'FOCUS': [100, 500, 1000],
      'LOWER_BOUND': [.4, .8, .9, .95],
      #'LOWER_BOUND': [.5, .7, .8, .9, .95],
      }
    full_sweep = get_options_product(OPTIONS, **kwargs)
    if n in range(50, 100):
      optionslist = [full_sweep[n]]
      print('Option', n)
    else:
      n_each = len(full_sweep)//MAXN  # This isn't right lol
      print(len(full_sweep), 'TOTAL')
      a, b = (n-1)*n_each, (n)*n_each
      if n == MAXN:
          b += 1
      optionslist = full_sweep[a:b]
      print(f'Option params: {a} - {b}')
  else:
    raise_ValueError('n invalid')

  ######
  # data = run_system(full_sweep[-1], 0, accept_input=False, n_sentences=n_sentences)
  # filename = 'data'+str(time.time()).replace('.', '_') + '.pickle'
  # fullfilename = os.path.join(FOLDER, filename)
  # with open(fullfilename, 'wb') as outfile:
  #   pickle.dump(data, outfile)
  # sys.exit()
  ######

  for i, O in enumerate(optionslist):
    print('-'*50)
    print(f'Running options: n={n}, i={i}')
    print()
    data = run_system(O, 0, accept_input=False,
                      n_sentences=n_sentences, nlp=nlp)
    filename = 'data'+str(time.time()).replace('.', '_') + '.pickle'
    fullfilename = os.path.join(FOLDER, filename)
    data['filename'] = filename
    print('pickling', fullfilename)
    with open(fullfilename, 'wb') as outfile:
      pickle.dump(data, outfile)

  # print(data)



