import pandas as pd
import numpy as np
import text_hammer as th
import re
import spacy
import nltk
from stop_words import get_stop_words
stopwords = get_stop_words('english')

class cleaning():
  def __init__(self):
    self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # disable parser, ner for faster loading
  
  # Define function for Lemmatization, remove stopword and feature selection using POS, spacy package
  def spacy_preprocess (self, text, lemma= True, pos= True, pos_select = ["VERB", "NOUN", "ADJ", "ADV", "PART"]):
    # Initialize spacy 'en_core_web_sm' model, keeping only tagger component needed for lemmatization
    # Parse the sentence using the loaded 'en' model object `nlp`
    doc = self.nlp(text)

    if pos== False:
      if lemma== True: text_preprocess= " ".join([token.lemma_.lower() for token in doc if not self.nlp.vocab[token.text].is_stop])
      if lemma== False: text_preprocess= " ".join([token.text.lower() for token in doc if not self.nlp.vocab[token.text].is_stop])
    else:
      if lemma== True : text_preprocess= " ".join([token.lemma_.lower() for token in doc if (token.pos_ in pos_select and not self.nlp.vocab[token.text].is_stop)])
      if lemma== False : text_preprocess= " ".join([token.text.lower() for token in doc if (token.pos_ in pos_select  and not self.nlp.vocab[token.text].is_stop)])
    # nlp.vocab[token.text].is_stop to remove stopwords
    return text_preprocess

  # function for removing stopword in a text
  # return text result after removing the stopwords
  def remove_stopword(self, text):
    wordList = []
    tempText = nltk.word_tokenize(text)
    for word in tempText:
      if word not in stopwords:
        wordList.append(word)
    text = (" ".join(wordList))
    return text

  def cleaning_process(self, df, column, expansion=False, rm_email=False, rm_html=False, rm_special=False, rm_accent=False, rm_regex=False, rm_stop=False, rm_stopbasic=False):
    if expansion:
      df[column] = df[column].apply(lambda x: th.cont_exp(x)) #you're -> you are; i'm -> i am
    if rm_email:
      df[column] = df[column].apply(lambda x: th.remove_emails(x))
    if rm_html:
      df[column] = df[column].apply(lambda x: th.remove_html_tags(x))

    if rm_special:
      df[column] = df[column].apply(lambda x: th.remove_special_chars(x))
    if rm_accent:
      df[column] = df[column].apply(lambda x: th.remove_accented_chars(x))

    if rm_regex:
      df[column] = df[column].apply(lambda x: re.sub(r'[^\w|^\']',' ',x))
      df[column] = df[column].apply(lambda x: re.sub("\d+", "",x))

    #df[column] = df[column].apply(lambda x: th.make_base(x)) #ran -> run, 
    #df[column] = df[column].apply(lambda x: th.spelling_correction(x).raw_sentences[0]) #seplling -> spelling
    if rm_stop:
      df[column] = df[column].apply(lambda x: self.spacy_preprocess(x, pos_select = ["VERB", "NOUN", "ADJ"]))

    if rm_stopbasic:
      df[column] = df[column].apply(lambda x: self.remove_stopword(x))

    return df