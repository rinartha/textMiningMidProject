from sklearn.decomposition import LatentDirichletAllocation
# Parameters tuning using Grid Search
from sklearn.model_selection import GridSearchCV
import numpy as np



class lda_clustering():

  # find the best LDA clustering method by employing gridsearch
  def auto_fit_transform(self, tfidf_matrix):
    # using gridsearch to find the number of component (topic) in dataset
    # min number of topic is 2 and max is 5
    grid_params = {'n_components' : list(range(2,5))}

    # LDA model
    lda_model = GridSearchCV(LatentDirichletAllocation(max_iter=1000),param_grid=grid_params)
    lda_model.fit(tfidf_matrix)

    # Best LDA model
    best_lda_model = lda_model.best_estimator_
    self.lda_result=best_lda_model.transform(tfidf_matrix)
    # print("Best LDA model's params" , lda_model.best_params_)
    # print("Best log likelihood Score for the LDA model",lda_model.best_score_)
    # print("LDA model Perplexity on train data", best_lda_model.perplexity(tfidf_matrix))
    return self.lda_result, best_lda_model, lda_model.best_params_['n_components']

  def fit_transform(self, number_of_topic, tfidf_matrix):
    # LDA model fitting for best parameter
    lda_model= LatentDirichletAllocation(n_components=number_of_topic,max_iter=1000,random_state=42, n_jobs=-1, verbose=0)
    #lda.fit(document_term_matrix)
    lda_model.fit(tfidf_matrix)
    self.lda_result=lda_model.transform(tfidf_matrix)
    return self.lda_result, lda_model, number_of_topic

  def assign_cluster(self, dataframe):
    cluster = list()
    for index in range (len(self.lda_result)):
      cluster.append(np.argmax(self.lda_result[index]))
    dataframe['cluster'] = cluster
    return dataframe