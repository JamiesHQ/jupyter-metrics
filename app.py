from happyfaces import happyfacer
from flask import Flask, render_template, request, redirect, send_from_directory
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, ranges, HoverTool
from bokeh.embed import components
from bokeh.layouts import row, column
import requests
import os
import datetime
import pandas as pd
import dill

app = Flask(__name__, static_url_path='')

@app.route('/')
def main():
  return redirect('/index')

#From DataCamp Intro to Layouts - Rows of Plots
# from bokeh.layouts import row
# layout = row(p1,p2,p3)
# output_file('row.html')
# show(layout)

@app.route('/static/<path:path>', methods=["GET"])
def static_path(path):
  return send_from_directory('static', path)

#GET request from issue_comments_jupyter_copy.csv
@app.route('/index', methods = ["GET"])
def index():
    if request.method =="GET":

        df = pd.read_csv('issue_comments_jupyter_copy.csv', sep=',')

        df['org'] = df['org'].astype('str')
        df['repo'] = df['repo'].astype('str')
        df['comments'] = df['comments'].astype('str')
        df['user'] = df['user'].astype('str')

        # https://stackoverflow.com/questions/39401481/how-to-add-data-labels-to-a-bar-chart-in-bokeh

        ####Start code for Plot1- Comments per Repo (in Jupyter Org) ####
        counts_per_repo = df.groupby(['org', 'repo'], as_index=False).count().sort_values('number')
        repos = counts_per_repo.repo.values
        numbers = counts_per_repo.number.values
        source = ColumnDataSource(dict(y=repos.tolist(), x=numbers.tolist()))
        plot1 = figure(x_axis_label='No. Comments', y_axis_label="Repository",  
                      title='Count of Jupyter GitHub Comments per Repo',
                      x_range=ranges.Range1d(start=0, 
                                             end=((counts_per_repo.number.max() + 1000) / 1000) * 1000),
                      y_range=source.data['y'],
                      )
        plot1.hbar(source=source, y='y', height=.50, right='x', left=0)

        # print "Created plot"
        #plot.hbar(data.index, data['Close'])
        #print "plot.line"
        #script, div = components(plot1)
        #return render_template('index.html', script=script, div=div)


        ####Start code for Plot2 Comments per User (in Jupyter Org)####
        counts_per_user = df.groupby(['user'], as_index=False).count().sort_values('number', ascending=False)
        user = counts_per_user.user.values
        numbers = counts_per_user.number.values
        source = ColumnDataSource(dict(y=user.tolist(), x=numbers.tolist()))

        plot2 = figure(x_axis_label='No. Comments', y_axis_label="User Name",  
              title='Count of Jupyter GitHub Comments per User (Top 50)',
              x_range=ranges.Range1d(start=0, 
                                     end=((counts_per_user.number.max() + 1000) / 1000) * 1000),
              y_range=source.data['y'][:50],
              )
        plot2.hbar(source=source, y='y', height=0.5, right='x', left=0)

        
        ####Start code for Plot3 TSNE Scatter Plot####
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

        p2column= column([plot2])
        r = row([plot1, p2column])
        #layout = column(p1,p2) - added below from https://campus.datacamp.com/courses/interactive-data-visualization-with-bokeh/layouts-interactions-and-annotations?ex=2
        script1, div1 = components(r)
        #script2, div2 = components(plot2)
        #script3, div3 = components(plot3)

        try:
          age, happyface = dill.load(open('happyface.dill', 'rb'))
        except:
          age = datetime.datetime.now() + datetime.timedelta(days=2)

        if (age - datetime.datetime.now()).total_seconds() > 60*60*24:
          happyface = happyfacer('issue_comments_jupyter_copy.csv')
          dill.dump((datetime.datetime.now(), happyface), open('happyface.dill', 'wb'))
        return render_template('index.html', script1=script1, div1=div1, happyface=happyface)



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
  app.run(port=33507, debug=True)


#Sources
#https://stackoverflow.com/questions/34853033/flask-post-the-method-is-not-allowed-for-the-requested-url

