from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, ranges
from bokeh.embed import components
import requests
import os
import pandas as pd
import seaborn as sns

app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

#Original GET request from ticker
@app.route('/index', methods = ["GET"])
def index():
    if request.method =="GET":
        #ticker = request.form["ticker"]
        #print "Ticker is:", ticker

        #data = Quandl.get("WIKI/%s" % ticker)
        df = pd.read_csv('issue_comments.csv', sep=',')
        #print "Got dataframe with", data.size, "elements"

        df['org'] = df['org'].astype('str')
        df['repo'] = df['repo'].astype('str')
        df['comments'] = df['comments'].astype('str')
        df['user'] = df['user'].astype('str')

        # https://stackoverflow.com/questions/39401481/how-to-add-data-labels-to-a-bar-chart-in-bokeh

        counts_per_repo = df.groupby(['org', 'repo'], as_index=False).count().sort_values('number')
        repos = counts_per_repo.repo.values
        numbers = counts_per_repo.number.values
        source = ColumnDataSource(dict(y=repos.tolist(), x=numbers.tolist()))
        plot = figure(x_axis_label='Counts', y_axis_label="Repo",  
                      title='Count of Jupyter GitHub Comments per Repo',
                      x_range=ranges.Range1d(start=0, 
                                             end=((counts_per_repo.number.max() + 1000) / 1000) * 1000),
                      y_range=source.data["y"],
                      )
        plot.hbar(source=source, y='y', height=.50, right='x', left=0)

        # print "Created plot"
        #plot.hbar(data.index, data['Close'])
        #print "plot.line"
        script, div = components(plot)
        return render_template('index.html', script=script, div=div)

#New get from GitHub API - comment out, do not run
# @app.route('/index', methods = ["GET", "POST"])
# def index():
#     if request.method =="GET":
#       return render_template('index.html')
#     elif request.method == "POST":
#         ticker = request.form["ticker"]
#         print "Ticker is:", ticker

#         data = Quandl.get("WIKI/%s" % ticker)
#         print "Got dataframe with", data.size, "elements"
#         plot = figure(
#               title='Jupyter GitHub Metrics',
#               x_axis_label='date',
#               x_axis_type='datetime')
#         print "Created plot"
#         plot.line(data.index, data['Close'])
#         print "plot.line"
#         script, div = components(plot)
#         return render_template('index.html', tickerout = request.form["ticker"], script=script, div=div)



if __name__ == '__main__':
  app.run(port=33507)


#Sources
#https://stackoverflow.com/questions/34853033/flask-post-the-method-is-not-allowed-for-the-requested-url

