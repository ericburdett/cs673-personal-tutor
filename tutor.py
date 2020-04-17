import time
from language import WordDist, LanguageModel, Evaluator
from user import UserKnowledge
from system import ReaderSystem
import numpy as np
import spacy

K = 50
MAX_SENTENCE_LENGTH = 60
NUM_SENTENCES = 25
KNOWN = 0
print(f'K: {K}')
print(f'MAX_SENTENCE_LENGTH: {MAX_SENTENCE_LENGTH}')
print(f'NUM_SENTENCES: {NUM_SENTENCES}')
print(f'KNOWN: {KNOWN}')

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

def main():
  print("Select a Topic:")
  print("[0]: News/Politics")
  print("[1]: Sports")
  print("[2]: Coronavirus")

  topic = -1
  while topic not in [0, 1, 2]:
    try:
        topic = int(input("Topic Number: "))
    except ValueError:
        print('Topic should be a number')

  print('\nLoading Language Learning System...')

  reader_system = ReaderSystem()

  nlp = spacy.load('en_core_web_md')

  lm = LanguageModel(k=K)
  lm.set_mask(np.ones(len(lm.tokenizer.get_vocab())))

  word_dist = WordDist()


  user_knowledge = UserKnowledge(word_dist.dict_normalized(), lm.tokenizer)

  sorted_words = sorted(
    ((k, v) for k, v in word_dist.dict_normalized().items()),
    key=lambda kv: kv[1], reverse=True)

  for i in range(KNOWN):
    user_knowledge.update([sorted_words[i][0]]*5, [True]*5)

  if topic == 0: # News/Politics
    prompts = NEWS_POLITICS_PROMPTS
    prompts_truncate = NEWS_POLITICS_TRUNCATE
    topic = "News"
  elif topic == 1: # Sports
    prompts = SPORTS_PROMPTS
    prompts_truncate = SPORTS_TRUNCATE
    topic = "Sports"
  elif topic == 2:
    prompts = CORONAVIRUS_PROMPTS
    prompts_truncate = CORONAVIRUS_TRUNCATE
    topic = "Health"
  else:
    raise ValueError('bad topic somehow')


  evaluator = Evaluator(topic, nlp)

  while True:
    print("Generating Sentence...")

    # Select a random prompt from above
    rand_index = np.random.randint(len(prompts))
    prompt = prompts[rand_index]
    should_truncate = prompts_truncate[rand_index]

    # Get/set the mask
    new_mask = user_knowledge.compute_mask()
    lm.set_mask(new_mask)

    # Get sentences from the language model
    sentences = lm.get_sentences(prompt, MAX_SENTENCE_LENGTH, NUM_SENTENCES)
    if should_truncate:
      sentences = [remove_prompt(sentence, prompt) for sentence in sentences]

    # Score the sentences based on evaluation criteria
    scores = evaluator.score_sentences(sentences)
    best_sentence = sentences[np.argmax(scores)]

    # Show the sentence to the user
    words, word_knowledge = reader_system.read_sentence(best_sentence)

    # Update our user knowledge and recompute the language model mask
    user_knowledge.update(words, word_knowledge)

if __name__ == "__main__":
  main()

  # lm = LanguageModel(k=K)
  # lm.set_mask(np.ones(len(lm.tokenizer.get_vocab())))

  # word_dist = WordDist()


  # user_knowledge = UserKnowledge(word_dist.dict_normalized(), lm.tokenizer)

  # sorted_words = sorted(
  #   ((k, v) for k, v in word_dist.dict_normalized().items()),
  #   key=lambda kv: kv[1], reverse=True)
  # for i in range(KNOWN):
  #   user_knowledge.update([sorted_words[i][0]]*5, [True]*5)
