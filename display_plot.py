import numpy as np
import random
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import nltk

class display_plot():
  def plot_wordcloud_for_cluster(self, mytitle, seg_list, mask_image, file_name):
    
    back_image = np.array(Image.open(mask_image))

    # List of colormaps
    colormaps = ['plasma', 'inferno', 'magma', 'viridis', 'cividis', 'twilight', 'tab10']

    # Randomly pick a colormap
    random_colormap = random.choice(colormaps)

    wc = WordCloud(
        width=1500,
        height=1500,
        background_color='white',               #   Background Color
        max_words=200,                    #   Max words
        mask=back_image,                       #   Background Image
        max_font_size=None,                   #   Font size
        # font_path="TaipeiSansTCBeta-Regular.ttf",
        random_state=50,                    #   Random color
        regexp=r"\w+(?:[-']\w+)*",  # Update the regexp parameter to include hyphens, you can mark out this line to hide the space character.
        contour_width=1,  # adjust the contour width
        contour_color='black',  # adjust the contour color
        colormap=random_colormap,  # choose a different colormap
        prefer_horizontal=0.9)                #   Ratio

    wc.generate(seg_list)
    
    wc.to_file(file_name)

  def plot_word_freq(self, top_k, title, text):
    data_list_word = nltk.word_tokenize(text)
    
    fdist = FreqDist(word for word in data_list_word)
    sortedWord = dict(sorted(fdist.items(), key=lambda tup: tup[1], reverse=True)[:top_k])
    #image = fdist.plot(top_k, cumulative=False, title=(title))
    # plt.show()
    return list(sortedWord.keys()), list(sortedWord.values())
    #plt.imsave(file_name, image)