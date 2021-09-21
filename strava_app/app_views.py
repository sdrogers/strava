
import requests
import logging

import pandas as pd
import json

from flask import (
    Blueprint, redirect, render_template, request, session, Response, url_for, make_response
)

from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired

import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("agg")

av = Blueprint('app_views', __name__, url_prefix='/app_views')

@av.route('/download')
def download():
    dates = json.loads(request.cookies.get('date'))
    average_hr = json.loads(request.cookies.get('average_heartrate'))
    min_per_km = json.loads(request.cookies.get('min_per_km'))
    df = pd.DataFrame(
        {
            'date': dates,
            'average_heartrate': average_hr,
            'min_per_km': min_per_km
        }
    )
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@av.route('/authenticate')
def authenticate():
    '''
    Do the initial token acquisition
    '''
    logging.info('Redirect to strava')
    return redirect("https://www.strava.com/oauth/authorize?client_id=22245&redirect_uri=http://localhost:5000/app_views/store_refresh&response_type=code&scope=activity:read_all")


def get_refresh_token(code):
    auth_url = "https://www.strava.com/oauth/token?client_id={your_client_id}&client_secret={your_client_secret}&code={your_code_from_previous_step}&grant_type=authorization_code".format(
           your_client_id=22245,
           your_client_secret="ae4e743a5e87baca1530ac16f59f941096b36536",
           your_code_from_previous_step=code)
    response = requests.post(auth_url)
    response_json = response.json()

    refresh_token = response_json['refresh_token']

    return refresh_token

@av.route('/store_refresh')
def store_refresh():
    '''
    Retrieve and store the token
    '''
    code = request.args.get('code')
    refresh_token = get_refresh_token(code)
    # store in cookies
    response = make_response(redirect(url_for('.get_access_token')))
    response.set_cookie('refresh_token', refresh_token, httponly=True)
    print(refresh_token)
    # Force retreival of access token
    return response

@av.route('/get_access_token')
def get_access_token():
    refresh_token = request.cookies.get('refresh_token')
    print(refresh_token)
    payload = {
        'client_id': "22245",
        'client_secret': 'ae4e743a5e87baca1530ac16f59f941096b36536',
        'refresh_token': refresh_token,
        'grant_type': "refresh_token",
        'f': 'json'
    }
    auth_url = "https://www.strava.com/oauth/token"
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()['access_token']
    response = make_response(redirect(url_for('.analysis_form')))
    response.set_cookie('access_token', access_token, httponly=True)
    return response

class InputForm(FlaskForm):
    recent_days = IntegerField('recent days', default=10, validators=[DataRequired()])
    date = DateField('start date', validators=[DataRequired()])
    max_hr = IntegerField('max_hr', default=145, validators=[DataRequired()])
    min_hr = IntegerField('min_hr', default=135, validators=[DataRequired()])

@av.route('/analysis_form', methods=['GET', 'POST'])
def analysis_form():
    access_token = request.cookies.get('access_token')
    refresh_token = request.cookies.get('refresh_token')

    form = InputForm()
    if form.validate_on_submit():
        return redirect(
            url_for('.analyse_data',
                date=form.date.data,
                recent_days=form.recent_days.data,
                max_hr=form.max_hr.data,
                min_hr=form.min_hr.data)
            )

    return render_template('analysis_form.html',
        access_token=access_token,
        refresh_token=refresh_token,
        form=form)

@av.route('/analyse_data')
def analyse_data():
    start_date = request.args.get('date')
    recent_days = request.args.get('recent_days')
    max_hr = request.args.get('max_hr')   
    min_hr = request.args.get('min_hr')


    return render_template('analyse_data.html',
        start_date=start_date,
        recent_days=recent_days,
        max_hr=max_hr,
        min_hr=min_hr
    )

@av.route('/make_plot')
def make_plot():
    start_date = request.args.get('date')
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    recent_days = int(request.args.get('recent_days'))
    max_hr = int(request.args.get('max_hr'))    
    min_hr = int(request.args.get('min_hr'))    
    access_token = request.cookies.get('access_token')
    runs = get_runs(start_date, max_hr, min_hr, access_token)

    import matplotlib.cm as cm
    import numpy as np
    from datetime import timedelta
    from sklearn.linear_model import LinearRegression, Lasso

    max_days = max(runs.days)
    n_bins = 1
    step = max_days / n_bins
    pred_x = [min(runs.average_heartrate), 
            max(runs.average_heartrate)]
    pred_x = np.asarray(pred_x)[:, None]

    colors = cm.Set1(np.linspace(0, 1, n_bins))

    fig = Figure(figsize = (10,10))
    axis = fig.add_subplot(2, 1, 1)
    pars = []
    coefs = []
    inters = []
    for i in range(n_bins):
        start_val = round(i*step)
        if i == 0 :
            start_val -= 1
        end_val = round((i+1)*step)
        temp = runs[runs.days > start_val]
        temp = temp[temp.days <= end_val]
        print(f'found {len(temp)}')
        x = np.asarray(temp.average_heartrate)[:,None]
        y = temp.min_per_km
        lr = LinearRegression()
        lr.fit(x, y)
        preds = lr.predict(pred_x)
        start_d = start_date + timedelta(start_val) 
        end_d = start_date + timedelta(end_val)
        label = f'{start_d.day}/{start_d.month} to {end_d.day}/{end_d.month}'
        axis.scatter(x, y, color = colors[i],
                    label = label)
        axis.plot(pred_x, preds, color = colors[i])
        coefs.append(lr.coef_[0])
        inters.append(lr.intercept_)
    axis.legend()
    axis.set_xlabel('Average heartrate')
    axis.set_ylabel('Average pace (min per km)')
    axis.grid()

    ax2 = fig.add_subplot(2, 1, 2)
    import GPy
    min_date = start_date
    X = np.array([(d - min_date).days for d in temp.nice_date])[:,None]
    y = temp.min_per_km.values[:,None]
    gpr = GPy.models.GPRegression(X, y)
    gpr.optimize()
    gpr.plot(ax=ax2)
    ax2.set_xlim(X[:,0].min(), X[:,0].max())
    ax2.set_ylim(0.9*y[:,0].min(), 1.05*y[:,0].max())
    ax2.set_xlabel('Day in the period')
    ax2.set_ylabel('Average pace (min per km)')
    ax2.grid()

    # highlight most recent n days
    max_day = max(runs.days)
    min_day = max_day - (recent_days - 1)
    temp = runs[runs.days >= min_day]
    axis.scatter(temp.average_heartrate, temp.min_per_km)
    a = temp.nice_date.values
    b = temp.average_heartrate.values
    c = temp.min_per_km.values
    
    for i, d in enumerate(a):
        ts = pd.to_datetime(str(d)) 
        e = ts.strftime('%d/%m')
        axis.text(b[i], c[i], e)


    

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    response = Response(output.getvalue(), mimetype='image/png')
    response.set_cookie('date', json.dumps([pd.to_datetime(str(v)).strftime('%d/%m/%y') for v in runs.nice_date.values]), httponly=True)
    response.set_cookie('min_per_km', json.dumps([v for v in runs.min_per_km.values]), httponly=True)
    response.set_cookie('average_heartrate', json.dumps([v for v in runs.average_heartrate.values]), httponly=True)

    return response


    return f"{start_date}, {recent_days}, {max_hr}, {len(runs)}"

def get_runs(start_date, max_hr, min_hr, access_token):
    activites_url = "https://www.strava.com/api/v3/athlete/activities"
    header = {'Authorization': 'Bearer ' + access_token}
    my_dataset = {}
    page_range = [1]
    for page_no in page_range:
        print(f'Getting page {page_no}')
        param = {'per_page': 200, 'page': page_no}
        my_dataset[page_no] = requests.get(activites_url, headers=header, params=param).json()
    activities = pd.json_normalize(my_dataset[1])
    if len(page_range) > 1:
        for page_no in page_range[1:]:
            temp = json_normalize(my_dataset[page_no])
            print(temp.start_date.iloc[0], temp.start_date.iloc[-1])
            activities = pd.concat([activities, temp])
            print(len(activities))

    runs = activities[activities.type.eq('Run')].copy()
    runs = filter_runs(runs, start_date, max_hr, min_hr)
    return runs

def filter_runs(runs, start_date, max_hr, min_hr):
    def date_format(row):
        nice_date = datetime.strptime(row['start_date'].split('T')[0], "%Y-%m-%d")
        return nice_date
    def m_per_hb(row):
        hb_per_s = row['average_heartrate'] / 60.
        m_per_s = row['average_speed']
        return m_per_s / hb_per_s
    def unix_time(row):
        return row['nice_date'].timestamp()
    def month_day(row):
        return f"{row['nice_date'].day}/{row['nice_date'].month}"
    def mins_per_km(row):
        m_per_s = row['average_speed']
        km_per_s = m_per_s / 1000.
        km_per_minute = km_per_s * 60.
        return 1. / (max(1e-6, km_per_minute))
        

    runs['nice_date'] = runs.apply(lambda row: date_format(row), axis=1)
    runs['nice_date'] = pd.to_datetime(runs['nice_date'])
    runs['m_per_hb'] = runs.apply(lambda row: m_per_hb(row), axis=1)
    runs['month_day'] = runs.apply(lambda row: month_day(row), axis=1)
    runs['min_per_km'] = runs.apply(lambda row: mins_per_km(row), axis=1)

    # min_date = datetime.strptime('2021-1-1', "%Y-%m-%d")
    min_speed = 2.2
    filtered_df = runs[runs['average_heartrate'] <= max_hr]
    print(f"Found {len(filtered_df)} rows")
    filtered_df = filtered_df[filtered_df['average_heartrate'] >= min_hr]
    print(f"Found {len(filtered_df)} rows")
    filtered_df = filtered_df[filtered_df['average_speed'] >= min_speed]
    print(f"Found {len(filtered_df)} rows")
    filtered_df = filtered_df[filtered_df['nice_date'] >= start_date]
    print(f"Found {len(filtered_df)} rows")

    # add days since cutoff
    filtered_df['days'] = (filtered_df['nice_date'] - start_date).dt.days
    filtered_df['month'] = filtered_df['nice_date'].dt.month


    filtered_df.sort_values('days', inplace=True)

    return filtered_df
