from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure
from bokeh.models.glyphs import HBar
from bokeh.embed import components
import requests
import os
import Quandl
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

		counts_per_repo = df.groupby(['org', 'repo']).count()

		plt.figure(figsize=(5,5))

		sns.countplot(y='repo', data=df, color ='c').set_title('Count of Jupyter GitHub Comments per Repo')
		plt.show()

		plot = figure(
              title='Count of Jupyter GitHub Comments per Repo',
              x_axis_label='count',
              x_axis_type='repo')
		# print "Created plot"
		#plot.hbar(data.index, data['Close'])
		plot = hbar(df, 'count', values='repo', title="Comments per Repo")
		#print "plot.line"
		script, div = components(plot)
		return render_template('index.html', script=script, div=div)

#New get from GitHub API - comment out, do not run
# @app.route('/index', methods = ["GET", "POST"])
# def index():
# 	if request.method =="GET":
# 	  return render_template('index.html')
# 	elif request.method == "POST":
# 		ticker = request.form["ticker"]
# 		print "Ticker is:", ticker

# 		data = Quandl.get("WIKI/%s" % ticker)
# 		print "Got dataframe with", data.size, "elements"
# 		plot = figure(
#               title='Jupyter GitHub Metrics',
#               x_axis_label='date',
#               x_axis_type='datetime')
# 		print "Created plot"
# 		plot.line(data.index, data['Close'])
# 		print "plot.line"
# 		script, div = components(plot)
# 		return render_template('index.html', tickerout = request.form["ticker"], script=script, div=div)



if __name__ == '__main__':
  app.run(port=33507)


#Sources
#https://stackoverflow.com/questions/34853033/flask-post-the-method-is-not-allowed-for-the-requested-url

