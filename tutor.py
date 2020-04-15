from language import WordDist, LanguageModel, Evaluator
from user import UserKnowledge
from system import ReaderSystem
import numpy as np
import spacy

K = 25
MAX_SENTENCE_LENGTH = 40
NUM_SENTENCES = 20

def remove_prompt(sentence, prompt):
  new_sentence = sentence.split(prompt)
  if len(new_sentence) < 2:
    return ''
  else:
    return sentence.split(prompt)[1]

def main():
  reader_system = ReaderSystem()

  nlp = spacy.load('en_core_web_md')

  lm = LanguageModel(k=K)
  lm.set_mask(np.ones(len(lm.tokenizer.get_vocab())))

  user_knowledge = UserKnowledge(WordDist().dict_normalized(), lm.tokenizer)

  topic = "News" # Add a user facing part that allows the user to select a topic...
  evaluator = Evaluator(topic, nlp)

  prompts = ['President Trump said']
  prompts_truncate = [False]


  while True:
    # Select a random prompt from above
    rand_index = np.random.randint(len(prompts))
    prompt = prompts[rand_index]
    should_truncate = prompts_truncate[rand_index]

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
    new_mask = user_knowledge.compute_mask()
    lm.set_mask(new_mask)
    

if __name__ == "__main__":
  main()