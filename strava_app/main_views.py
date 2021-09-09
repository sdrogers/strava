from flask import (
    Blueprint, redirect, render_template, request, session, Response
)
import requests

import io

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("TKAgg")

bp = Blueprint('main_views', __name__, url_prefix='/main_views')

@bp.route('/start', methods=['GET'])
def start():
    return redirect("https://www.strava.com/oauth/authorize?client_id=22245&redirect_uri=http://localhost:5000/main_views/back&response_type=code&scope=activity:read_all")

@bp.route('/back', methods=['GET'])
def back():
    code = request.args.get('code')
    auth_url = "https://www.strava.com/oauth/token?client_id={your_client_id}&client_secret={your_client_secret}&code={your_code_from_previous_step}&grant_type=authorization_code".format(
           your_client_id=22245,
           your_client_secret="ae4e743a5e87baca1530ac16f59f941096b36536",
           your_code_from_previous_step=code)
    print(auth_url) 
    response = requests.post(auth_url)
    response_json = response.json()
    print(response_json)
    payload = {
        'client_id': "22245",
        'client_secret': 'ae4e743a5e87baca1530ac16f59f941096b36536',
        'refresh_token': response_json['refresh_token'],
        'grant_type': "refresh_token",
        'f': 'json'
    }
    auth_url = "https://www.strava.com/oauth/token"
    print("Requesting Token...\n")
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()['access_token']
    print(res)
    
    redirect_url = f"/main_views/analysis?access_token={access_token}"
    return redirect(redirect_url)

@bp.route('/analysis', methods=['GET'])
def analysis():
    activites_url = "https://www.strava.com/api/v3/athlete/activities"
    access_token = request.args.get('access_token')
    header = {'Authorization': 'Bearer ' + access_token}
    import pandas as pd
    from pandas import json_normalize
    my_dataset = {}
    page_range = [1]
    for page_no in page_range:
        print(f'Getting page {page_no}')
        param = {'per_page': 200, 'page': page_no}
        my_dataset[page_no] = requests.get(activites_url, headers=header, params=param).json()
    activities = json_normalize(my_dataset[1])
    if len(page_range) > 1:
        for page_no in page_range[1:]:
            temp = json_normalize(my_dataset[page_no])
            print(temp.start_date.iloc[0], temp.start_date.iloc[-1])
            activities = pd.concat([activities, temp])
            print(len(activities))

    runs = activities[activities.type.eq('Run')].copy()

    from datetime import datetime
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

    min_date = datetime.strptime('2021-1-1', "%Y-%m-%d")
    max_hr = 190.0
    min_speed = 2.2
    filtered_df = runs[runs['average_heartrate'] <= max_hr]
    print(f"Found {len(filtered_df)} rows")
    filtered_df = filtered_df[filtered_df['average_speed'] >= min_speed]
    print(f"Found {len(filtered_df)} rows")
    filtered_df = filtered_df[filtered_df['nice_date'] >= min_date]
    print(f"Found {len(filtered_df)} rows")

    # add days since cutoff
    filtered_df['days'] = (filtered_df['nice_date'] - min_date).dt.days
    filtered_df['month'] = filtered_df['nice_date'].dt.month


    filtered_df.sort_values('days', inplace=True)

    import matplotlib.cm as cm
    import numpy as np
    from datetime import timedelta
    from sklearn.linear_model import LinearRegression, Lasso

    max_days = max(filtered_df.days)
    n_bins = 3
    step = max_days / n_bins
    pred_x = [min(filtered_df.average_heartrate), 
            max(filtered_df.average_heartrate)]
    pred_x = np.asarray(pred_x)[:, None]

    colors = cm.Set1(np.linspace(0, 1, n_bins))

    fig = Figure(figsize = (10,6))
    axis = fig.add_subplot(1, 1, 1)
    pars = []
    coefs = []
    inters = []
    for i in range(n_bins):
        start_val = round(i*step)
        if i == 0 :
            start_val -= 1
        end_val = round((i+1)*step)
        temp = filtered_df[filtered_df.days > start_val]
        temp = temp[temp.days <= end_val]
        print(f'found {len(temp)}')
        x = np.asarray(temp.average_heartrate)[:,None]
        y = temp.min_per_km
        lr = LinearRegression()
        lr.fit(x, y)
        preds = lr.predict(pred_x)
        start_d = min_date + timedelta(start_val) 
        end_d = min_date + timedelta(end_val)
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
    # highlight most recent n days
    n = 10
    max_day = max(filtered_df.days)
    min_day = max_day - (n - 1)
    temp = filtered_df[filtered_df.days >= min_day]
    axis.scatter(temp.average_heartrate, temp.min_per_km)
    a = temp.nice_date.values
    b = temp.average_heartrate.values
    c = temp.min_per_km.values
    
    for i, d in enumerate(a):
        ts = pd.to_datetime(str(d)) 
        e = ts.strftime('%d/%m')
        axis.text(b[i], c[i], e)

    print(max(temp.total_elevation_gain))

    

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

