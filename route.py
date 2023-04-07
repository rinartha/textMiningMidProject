from flask import render_template, request, redirect
from midtermProject import app
from midtermProject.model_clustering import *
from midtermProject.cleaning import *
from midtermProject.vectorizer import *
from midtermProject.display_plot import *
from midtermProject.ngram import *
import pandas as pd
import os

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['dataset']
        global df
        df = pd.read_csv(uploaded_file, sep=",", header=0)
        return render_template("clustering.html", 
        tables=[df.to_html(classes='table table-striped table-responsive table-bordered', border=0, table_id="data_table", justify="left")],
        titles=df.columns.values, 
        dataForm="",
        number_of_cluster = 0,
        word_freq = [],
        number_word = [],
        word_cloud = False,
        word_frequency = False)
    else:
        return render_template('index.html')

def boolean_input(var_input):
    if var_input == "1":
        return True
    return False

@app.route('/clustering', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        column_name = request.form.get('column')
        expansion = request.form.get('expansion')
        rm_email = request.form.get('rm_email')
        rm_html = request.form.get('rm_html')
        rm_special = request.form.get('rm_special')
        rm_accent = request.form.get('rm_accent')
        rm_regex = request.form.get('rm_regex')
        rm_stop = request.form.get('rm_stop')
        method_vectorizer = request.form.get('vectorizer')
        clustering = request.form.get('clustering')
        selectImage = request.form.get('selectImage')
        word_frequency = request.form.get('word_frequency')
        word_cloud = request.form.get('word_cloud')
        num_of_cluster = request.form.get('num_of_cluster')
        rm_stopbasic = request.form.get('rm_stopbasic')

        dataForm = [column_name, expansion, rm_email, rm_html,
                    rm_special , rm_accent, rm_regex, rm_stop,
                    method_vectorizer, clustering, selectImage, word_frequency, 
                    word_cloud, num_of_cluster, rm_stopbasic]

        pre_processing = cleaning()

        cleaning_df = pre_processing.cleaning_process(df.copy(), column_name, boolean_input(expansion), 
                    boolean_input(rm_email), boolean_input(rm_html),
                    boolean_input(rm_special), boolean_input(rm_accent), 
                    boolean_input(rm_regex), boolean_input(rm_stop), boolean_input(rm_stopbasic))


        mask_image = "./static/image/"+str(selectImage)+".png"
        ngram = custom_ngram()
        unigram = ngram.ngram_form_text(cleaning_df[column_name], 1)
        bigram = ngram.ngram_form_text(cleaning_df[column_name], 2)
        trigram = ngram.ngram_form_text(cleaning_df[column_name], 3)
        file_name_1 = "./static/image_result/word_n_gram_1.png"
        file_name_2 = "./static/image_result/word_n_gram_2.png"
        file_name_3 = "./static/image_result/word_n_gram_3.png"
        ngram_plot = display_plot()
        ngram_plot.plot_wordcloud_for_cluster("Unigram", unigram, mask_image, file_name_1)
        ngram_plot.plot_wordcloud_for_cluster("Bigram", bigram, mask_image, file_name_2)
        ngram_plot.plot_wordcloud_for_cluster("Trigram", trigram, mask_image, file_name_3)

        tfidf = vectorizer(cleaning_df.copy(), column_name)

        if clustering == "lda":
            lda = lda_clustering()
            if num_of_cluster == "auto":
                lda_model_result, lda_model, number_of_cluster = lda.auto_fit_transform(tfidf.get_term_matrix())
            else:
                lda_model_result, lda_model, number_of_cluster = lda.fit_transform(int(num_of_cluster), tfidf.get_term_matrix())
            
            new_df = lda.assign_cluster(cleaning_df.copy())
        elif clustering == "kmeans":
            kmeans = kmeans_clustering()
            if num_of_cluster == "auto":
                kmeans_model_result, kmeans_model, number_of_cluster = kmeans.auto_k_means(tfidf.get_term_matrix())
            else:
                kmeans_model_result, kmeans_model, number_of_cluster = kmeans.k_means(int(num_of_cluster), tfidf.get_term_matrix())
            
            new_df = kmeans.assign_cluster(cleaning_df.copy())
        else:
            kmedoid = kmedoid_clustering()
            if num_of_cluster == "auto":
                kmedoid_model_result, kmedoid_model, number_of_cluster = kmedoid.auto_k_medoids(tfidf.get_term_matrix())
            else:
                kmedoid_model_result, kmedoid_model, number_of_cluster = kmedoid.k_medoids(int(num_of_cluster), tfidf.get_term_matrix())
            
            new_df = kmedoid.assign_cluster(cleaning_df.copy())

        word_freq = []
        number_word = []
        for cluster_index in range(number_of_cluster):
            this_wordlist = ''
            title = f"Word for Cluster {cluster_index+1}"
            file_name_wc = "./static/image_result/word_cloud_"+str(int(cluster_index)+1)+".png"
            this_wordlist = new_df[new_df['cluster'] == cluster_index][column_name]
            this_seg_list=' '.join(this_wordlist) # convert list into string seperated with space character
            wc = display_plot()
            if boolean_input(word_cloud):
                wc.plot_wordcloud_for_cluster(title, this_seg_list, mask_image, file_name_wc)
            
            if boolean_input(word_frequency):
                word_freq_list, number_word_list = wc.plot_word_freq(30, title, this_seg_list)
                word_freq.append(word_freq_list)
                number_word.append(number_word_list)

        return render_template("clustering.html", 
        tables=[new_df.to_html(classes='table table-striped table-responsive table-bordered', border=0, table_id="data_table", justify="left")], 
        titles=df.columns.values, 
        dataForm=dataForm,
        number_of_cluster = number_of_cluster,
        word_freq = word_freq,
        number_word = number_word,
        word_cloud = boolean_input(word_cloud),
        word_frequency = boolean_input(word_frequency))
    