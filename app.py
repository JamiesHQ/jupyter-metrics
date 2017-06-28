from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure
from bokeh.embed import components
import requests
import os
import Quandl

app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index', methods = ["GET", "POST"])
def index():
	if request.method =="GET":
	  return render_template('index.html')
	elif request.method == "POST":
		ticker = request.form["ticker"]
		print "Ticker is:", ticker

		data = Quandl.get("WIKI/%s" % ticker)
		print "Got dataframe with", data.size, "elements"
		plot = figure(
              title='Jupyter GitHub Metrics',
              x_axis_label='date',
              x_axis_type='datetime')
		print "Created plot"
		plot.line(data.index, data['Close'])
		print "plot.line"
		script, div = components(plot)
		return render_template('index.html', tickerout = request.form["ticker"], script=script, div=div)

if __name__ == '__main__':
  app.run(port=33507)


#Sources
#https://stackoverflow.com/questions/34853033/flask-post-the-method-is-not-allowed-for-the-requested-url

