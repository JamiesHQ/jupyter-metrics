import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


## below from http://t-redactyl.io/blog/2017/04/applying-sentiment-analysis-with-vader-and-the-twitter-api.html
def happyfacer(df):

    analyzer = SentimentIntensityAnalyzer()
    vs_comment = []
    vs_compound = []
    vs_pos = []
    vs_neu = []
    vs_neg = []
    vs_repo = []
    vs_user = []

    for i in range(0, len(df.comments)):
        vs_user.append(df.user[i])
        vs_repo.append(df.repo[i])
        vs_comment.append(df.comments[i])
        vs_compound.append(analyzer.polarity_scores(df.comments[i])['compound'])
        vs_pos.append(analyzer.polarity_scores(df.comments[i])['pos'])
        vs_neu.append(analyzer.polarity_scores(df.comments[i])['neu'])
        vs_neg.append(analyzer.polarity_scores(df.comments[i])['neg'])

    sentiment_df = pd.DataFrame({
                            'User': vs_user,
                            'Repo': vs_repo,
                            'Comment': vs_comment,
                            'Compound': vs_compound,
                            'Positive': vs_pos,
                            'Neutral': vs_neu,
                            'Negative': vs_neg})
    sentiment_df = sentiment_df[['User','Repo','Comment', 'Compound', 'Positive', 'Neutral', 'Negative']]

    return sentiment_df.groupby('Repo')[['Compound']].mean()



