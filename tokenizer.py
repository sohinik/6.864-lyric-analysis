import re

class Tokenizer:
  def __init__(self, min_occur=10):
    self.word_to_token = {}
    self.token_to_word = {}
    self.word_count = {}

    self.word_to_token['<unk>'] = 0
    self.token_to_word[0] = '<unk>'
    self.vocab_size = 1

    self.min_occur = min_occur

  def fit(self, corpus):
    for song in corpus:
      song = song.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", song)
      for word in words:
          if word not in self.word_count:
              self.word_count[word] = 0
          self.word_count[word] += 1

    for song in corpus:
      song = song.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", song)
      for word in words:
        if self.word_count[word] < self.min_occur:
          continue
        if word in self.word_to_token:
          continue
        self.word_to_token[word] = self.vocab_size
        self.token_to_word[self.vocab_size] = word
        self.vocab_size += 1

  def tokenize(self, corpus):
    tokenized_corpus = []
    for song in corpus:
      song = song.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", song)
      tokenized_song = []
      for word in words:
        if word not in self.word_to_token:
          tokenized_song.append(0)
        else:
          tokenized_song.append(self.word_to_token[word])
      tokenized_corpus.append(tokenized_song)
    return tokenized_corpus

  def de_tokenize(self, tokenized_corpus):
    corpus = []
    for tokenized_song in tokenized_corpus:
      song = []
      for token in tokenized_song:
        song.append(self.token_to_word[token])
      corpus.append(" ".join(song))
    return corpus
