from sklearn.feature_extraction.text import TfidfVectorizer

class vectorizer():
  def __init__(self, dataframe, column):
    self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,3), use_idf=True, smooth_idf=True)
    self.document_term_matrix = self.vectorizer.fit_transform(dataframe[column].tolist()).astype(float)
    # print ("document matrix shape :", self.document_term_matrix.shape)

  def get_vocab(self):
    return self.vectorizer.get_feature_names_out()

  def get_term_matrix(self):
    return self.document_term_matrix