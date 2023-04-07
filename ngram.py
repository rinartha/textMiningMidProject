from nltk import ngrams
import nltk

class custom_ngram():
  def ngram_form_text(self, list_of_text, n):
    text=' '.join(list_of_text)
    ngrams_result = ngrams(nltk.word_tokenize(text), n)
    joined_ngram = []
    ngrams_list =[gram for gram in ngrams_result]
    for ngram in ngrams_list:
        joined_ngram.append('_'.join(ngram))
    result_text = ' '.join(joined_ngram)
    return result_text