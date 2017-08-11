
# coding: utf-8

# # Jupyter Comment Clusters
# ### (K-Means Clustering)

# In[2]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from string import maketrans

def get_comment_clusters(df):


  import lda
  from sklearn.feature_extraction.text import CountVectorizer

  n_topics = 20 # number of topics
  n_iter = 500 # number of iterations
  #circling back to add stop words
  stopwords = text.ENGLISH_STOP_WORDS.union(['want', 'need', 'python', 'ipython', 'use', 'using', 'user',
                                             'way', 'cell', 'kernel', 'think','git','jupyter', 'notebook', 
                                             'https', 'github', 'com', 'html', 'http', 'org','ellisonbg','don',
                                             'brian', 'granger', 'things', 'like', 'thanks', 'just', 'tried', 'minrk',
                                            'issue', 'issuecomment'])


  for repo, repo_df in df.groupby('repo'):
      if repo == 'notebook':
          # vectorizer: ignore English stopwords & words that occur less than 5 times
          all_comments = []
          for issue, comment_df in repo_df.groupby('number'):
              full_comment = ' '.join(comment_df['comments'].values)
              #chars_to_remove = [chr(x) for x in range(32)]
              #chars_to_remove.extend(['<', '>', '{', '}', '&'])
              #trans = ['\0' for x in range(len(chars_to_remove))]
              #t = maketrans(''.join(chars_to_remove), ''.join(trans))
              #full_comment = full_comment.translate(t)
              all_comments.append(' '.join(comment_df['comments'].values))

          cvectorizer = CountVectorizer(min_df=5, max_df=0.5, stop_words=stopwords)
          cvz = cvectorizer.fit_transform(all_comments)
          #vocab = {v: k for k, v in cvectorizer.vocabulary_.iteritems()}
          vocab = cvectorizer.get_feature_names()
          # train an LDA model
          lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
          #lda_model = models.LdaModel( matutils.Sparse2Corpus(cvz, documents_columns=False), num_topics=n_topics, passes=n_iter, id2word = vocab)
          X_topics = lda_model.fit_transform(cvz)
          break

  #lda_model.print_topics(num_topics=5, num_words=5)



  # Most common words in clusters. Read through the documents to assign meaningful cluster names and label the clusters. 

  # ### TSNE

  from sklearn.manifold import TSNE

  # a t-SNE model
  # angle value close to 1 means sacrificing accuracy for speed
  # pca initializtion usually leads to better results 
  tsne_model = TSNE(n_components=2, verbose=1, random_state=42, angle=.8, init='pca', method='exact')

  # 20-D -> 2-D
  tsne_lda = tsne_model.fit_transform(X_topics)


  # In[49]:

  import numpy as np
  import bokeh.plotting as bp
  from bokeh.plotting import save
  from bokeh.models import HoverTool

  n_top_words = 5 # number of keywords we show

  # 20 colors
  colormap = np.array([
      "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
      "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
      "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
      "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
  ])


  # In[51]:

  _lda_keys = []
  for i in xrange(X_topics.shape[0]):
    _lda_keys +=  X_topics[i].argmax(),


  # In[52]:

  topic_summaries = []
  topic_word = lda_model.topic_word_  # all topic words
  vocab = cvectorizer.get_feature_names()
  for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
    topic_summaries.append(' '.join(topic_words)) # append!


  # In[61]:

  title = 'Jupyter Comments'
  num_example = 300#len(X_topics)

  plot_lda = bp.figure(plot_width=800, plot_height=600,
                       title=title,
                       tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                       x_axis_type=None, y_axis_type=None, min_border=1)

  plot_lda.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
                   color=colormap[_lda_keys][:num_example],
                   source=bp.ColumnDataSource({
                     "content": all_comments[:num_example],
                     "topic_key": _lda_keys[:num_example]
                     }))

  # randomly choose a news (within a topic) coordinate as the crucial words coordinate
  topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
  for topic_num in _lda_keys:
    if not np.isnan(topic_coord).any():
      break
    topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

  # plot crucial words
  for i in xrange(X_topics.shape[1]):
    plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

  # hover tools
  hover = plot_lda.select(dict(type=HoverTool))
  hover.tooltips = {"content": "@content - topic: @topic_key"}

  # save the plot
  #save(plot_lda, '{}.html'.format(title))
  #show(plot_lda)
  return plot_lda

#