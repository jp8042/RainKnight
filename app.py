from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


application = app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predictmonth',methods=['POST'])
def predictmonth():
  print ('Hello Month')
  dataset=pd.read_csv('rainfall.csv')
  #y = dataset.iloc[:116, 5:6].values
  #X = dataset.iloc[:116, 0:1].values
  X=dataset.iloc[3888:4003,1:2].values
  y=dataset.iloc[3888:4003,14:15].values

  #from sklearn.preprocessing import StandardScaler
  #sc_X = StandardScaler()
  #sc_y = StandardScaler()
  #X = sc_X.fit_transform(X)
  #y = sc_y.fit_transform(y)

  from sklearn.svm import SVR
  from sklearn.model_selection import GridSearchCV
  #regressor =SVR(kernel = 'rbf')
  tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5,1e-6,1e-7,1e-8],
  					 'C': [1,2,3,4,5,6,7,8, 10, 100, 1000]},
  					{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

  regressor = GridSearchCV(SVR(), tuned_parameters, cv=5)
  regressor.fit(X, y)
  print ('Hello Month2')
  if request.method == 'POST':
    print ('Hello Month3')
    comment = request.form['predictmonth']
    data = str([comment][0])
    year = data[:4]
    month = float(data[5:7])
    date = data[8:]
    #   data is list, extract and convert into string format and get 1st four letters.
    if month == 1:
      my_label = 'January'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,2:3].values
    elif month == 2:
      my_label = 'Feburary'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,3:4].values
    elif month == 3:
      my_label = 'March'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,4:5].values
    elif month == 4:
      my_label = 'April'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,5:6].values
    elif month == 5:
      my_label = 'May'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,6:7].values
    elif month == 6:
      my_label = 'June'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,7:8].values
    elif month == 7:
      my_label = 'July'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,8:9].values
    elif month == 8:
      my_label = 'August'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,9:10].values
    elif month == 9:
      my_label = 'September'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,10:11].values
    elif month == 10:
      my_label = 'October'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,11:12].values
    elif month == 11:
      my_label = 'Novenber'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,12:13].values
    else:
      my_label = 'December'
      X=dataset.iloc[3888:4003,1:2].values
      y=dataset.iloc[3888:4003,13:14].values

    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    #regressor =SVR(kernel = 'rbf')
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5,1e-6,1e-7,1e-8],
                     'C': [1,2,3,4,5,6,7,8, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    regressor = GridSearchCV(SVR(), tuned_parameters, cv=5)
    regressor.fit(X, y)


    #regressor.fit(X, y)
    my_prediction = regressor.predict((float(year)))
    my_label = ' ' + my_label
    print ('Hello Month')
    print(my_label)
    return render_template('result.html',prediction = float("{0:.2f}".format(my_prediction[0])), label = my_label)


@app.route('/predictyear',methods=['POST'])
def predictyear():
  print ('Hello Year')
  dataset=pd.read_csv('rainfall.csv')
  #y = dataset.iloc[:116, 5:6].values
  #X = dataset.iloc[:116, 0:1].values
  X=dataset.iloc[3888:4003,1:2].values
  y=dataset.iloc[3888:4003,14:15].values

  #from sklearn.preprocessing import StandardScaler
  #sc_X = StandardScaler()
  #sc_y = StandardScaler()
  #X = sc_X.fit_transform(X)
  #y = sc_y.fit_transform(y)

  from sklearn.svm import SVR
  from sklearn.model_selection import GridSearchCV
  #regressor =SVR(kernel = 'rbf')
  tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5,1e-6,1e-7,1e-8],
  					 'C': [1,2,3,4,5,6,7,8, 10, 100, 1000]},
  					{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

  regressor = GridSearchCV(SVR(), tuned_parameters, cv=5)
  regressor.fit(X, y)
  
  if request.method == 'POST':
    comment = request.form['predictyear']
    data = str([comment][0])
    year = data[:4]
    month = data[5:7]
    date = data[8:]
    #   data is list, extract and convert into string format and get 1st four letters.
    my_prediction = regressor.predict((float(year)))
    my_label = 'annual year ' + year
    print(my_label)
    return render_template('result.html',prediction = float("{0:.2f}".format(my_prediction[0])), label = my_label)
    #sending data value to check the format of date stored in the list



@app.route('/predictmonsoon',methods=['POST'])
def predictmonsoon():
  dataset= pd.read_csv("rainfall.csv")

  #X, y = make_regression(n_features=1, n_informative=2,random_state=0, shuffle='FALSE')

  
  
  if request.method == 'POST':
    print ('Hello Why')
    comment = request.form['predictmonsoon']
    get_month = request.form['selectmonth']
    data = float([comment][0])
    month = get_month
    if month == "January":
      X=dataset.iloc[3888:4003,2:3].values
      y=dataset.iloc[3888:4003,17:18].values
    elif month == "Feburary":
      X=dataset.iloc[3888:4003,3:4].values
      y=dataset.iloc[3888:4003,17:18].values
    elif month == "March":
      X=dataset.iloc[3888:4003,4:5].values
      y=dataset.iloc[3888:4003,17:18].values
    elif month == "April":
      X=dataset.iloc[3888:4003,5:6].values
      y=dataset.iloc[3888:4003,17:18].values
    else : 
      X=dataset.iloc[3888:4003,6:7].values
      y=dataset.iloc[3888:4003,17:18].values
          
    #   data is list, extract and convert into string format and get 1st four letters.
    regr = RandomForestRegressor(max_depth=30, random_state=5,n_estimators=200)
    regr.fit(X, y)

    my_prediction = regr.predict(data)
    my_label = 'monsoon (June - September), given month ' + get_month 
    print(my_label)
    return render_template('result.html',prediction = float("{0:.2f}".format(my_prediction[0])), label = my_label)
  
  
if __name__ == '__main__':
	app.run(debug=True)