
# coding: utf-8

# # Jupyter Comment Clusters
# ### (K-Means Clustering)

# In[2]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:

df = pd.read_csv('issue_comments_jupyter_copy.csv')
df['org'] = df['org'].astype('str')
df['number'] = df['number'].astype('str')
df['repo'] = df['repo'].astype('str')
df['comments'] = df['comments'].astype('str')
df['user'] = df['user'].astype('str')


# Add stopwords into the vectorizer and bring max_df down to focus on the center of the cluster

# In[40]:

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


# In[47]:

#lda_model.print_topics(num_topics=5, num_words=5)
X_topics.shape
np.version.version



# Most common words in clusters. Read through the documents to assign meaningful cluster names and label the clusters. 

# ### TSNE

# In[48]:

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
num_example = 50 #len(X_topics)

plot_lda = bp.figure(plot_width=800, plot_height=600,
                     title=title,
                     #tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

plot_lda.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
                 color=colormap[_lda_keys][:num_example],
                 source=bp.ColumnDataSource({
                   "content": all_comments[:num_example],
                   "topic_key": _lda_keys[:num_example]
                   }))


# In[63]:

from bokeh.io import output_notebook, show
output_notebook()

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
#hover = plot_lda.select(dict(type=HoverTool))
#hover.tooltips = {"content": "@content - topic: @topic_key"}

# save the plot
#save(plot_lda, '{}.html'.format(title))
show(plot_lda)


# In[22]:

from bokeh.io import output_notebook, show
output_notebook()


# In[23]:

from sklearn.manifold import TSNE
xycoords = TSNE().fit_transform(X)


# In[24]:

X.head()


# In[25]:

xycoords.shape


# In[26]:

comments.shape


# In[27]:

colormap = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", 
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", 
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", 
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
]


# In[28]:

import numpy as np


# In[29]:

import pandas as pd


# In[30]:

df = pd.DataFrame(xycoords)
df = df.rename(columns={0: 'x', 1: 'y', 2:'terms'})
df.head()


# Read this to move forward: https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html

# In[31]:

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

source = ColumnDataSource(df)
hover = HoverTool(tooltips=[("term", "@term")])

plot3 = figure(
    plot_width=900,
    plot_height=700,
    title='Comments Map by t-SNE',
    tools='pan,wheel_zoom,box_zoom,reset,hover,previewsave'
    )

#plot.add_tools(hover)

plot3.circle(x='x', y='y', color='blue', source=source)

show(plot3)


# In[ ]:




# In[ ]:




# ## Import data from API call

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


# In[30]:

df = pd.read_csv('issue_comments_jupyter_copy.csv')

df['org'] = df['org'].astype('str')
df['repo'] = df['repo'].astype('str')
df['comments'] = df['comments'].astype('str')
df['user'] = df['user'].astype('str')


# ## Load data into a sparse matrix

# In[31]:

struct_comments = df.comments.to_sparse()


# In[32]:

struct_comments.shape


# In[34]:

struct_comments.head()


# - We want a matrix: rows are comments, columns are terms used
#     - cells will be 1 if word exists in that comment, 0 if not
# - This code takes <1 second to run 

# In[6]:

get_ipython().run_cell_magic(u'time', u'', u'\nuser_ids = []\ncomment_ids = []\ncomment_to_id = {}\ni = 0\nwith open (\'issue_comments_jupyter_copy.csv\', \'r\') as f: \n    for line in f:\n        for item in line.rstrip().split(\',\')[1:]:\n            if item not in comment_to_id:\n                comment_to_id[item] = len(comment_to_id)\n            user_ids.append(i)\n            comment_ids.append(comment_to_id[item])\n        i+=1\n        \nimport numpy as np\nfrom scipy.sparse import csr_matrix #https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.sparse.csr_matrix.html\n\nrows = np.array(comment_ids)\ncols = np.array(user_ids)\ndata = np.ones((len(user_ids),))\nnum_rows = len(comment_to_id)\nnum_cols = i\n\n#the code above exists to feed this call\nadj = csr_matrix( (data, (rows,cols)), shape=(num_rows, num_cols) )\nprint adj.shape\nprint ""\n\n# now we have our matrix, so let\'s gather up a bit of info about it\nusers_per_issue = adj.sum(axis=1).A1\ncomments = range(len(comment_to_id))\nfor item in comment_to_id:\n    comments[comment_to_id[item]]=item\ncomments = np.array(comments)\n            ')


# # Unwieldy data
# 
# Our adjacency matrix is a bit problematic to deal with as-is:
# 
# * It's a very wide: 850,000 columns
# * It's very sparse: only about 0.06% full
# * It's a binary matrix: only 0's and 1's
# 
# ## Dimensionality reduction
# 
# * Family of algorithms for solving this problem
# * AKA decomposition, compression, feature extraction
#     * it's to big matrices what JPEG is to photos, MP3 to music, MPEG for video etc
# * The output will be as wide as we ask for
#     * The wider it is the less lossy the compression will be
# * The output matrix will be dense
# * The output matrix will have continuous real values
# * `scikit-learn` has a [decomposition package](http://scikit-learn.org/stable/modules/decomposition.html)
# 
# # We'll use `TruncatedSVD`
# 
# * [TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) from `scikit-learn`
# * SVD stands for Singular Vector Decomposition, Truncated because we only want part of the computation
# * Good mathy description of how it works in [Chapter 11](http://infolab.stanford.edu/~ullman/mmds/ch11.pdf) of online book [Mining of Massive Datasets](http://www.mmds.org/)
# * Right now let's focus on how to use it and exploring the output

# In[19]:

get_ipython().run_cell_magic(u'time', u'', u"\nfrom sklearn.decomposition import TruncatedSVD \nfrom sklearn.preprocessing import normalize \n\nsvd = TruncatedSVD(n_components=100)\nembedded_coords = normalize(svd.fit_transform(adj), norm='l1')\nprint embedded_coords.shape")


# The output is kind of neat:
# 
# * Each row is like a set of coordinates in a 100-dimensional space for a subreddit
# * Each column defines one axis of this 100-dimensional space, ordered by how much information they capture
# * We can look at how much of the original matrix we captured with the first N dimensions
# * The first 2 capture around 25%
# * The 100 we will use capture around 60%

# In[20]:

get_ipython().magic(u'matplotlib inline')
pd.DataFrame(np.cumsum(svd.explained_variance_ratio_)).plot(figsize=(13, 8))


# In[28]:

# this function will show you the axes on which a particular subreddit scores the highest/lowest
def pickOutComment(item):
    sorted_axes = embedded_coords[list(comments).index(item)].argsort()[::-1]
    return pd.DataFrame(comments[np.argsort(embedded_coords[:,sorted_axes], axis=0)[::-1]], columns=sorted_axes)

pickOutComment("notebook")


# In[ ]:



