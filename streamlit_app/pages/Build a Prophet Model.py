import streamlit as st

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from prophet.serialize import model_to_json, model_from_json


st.title("Time Series Forecasting Using Prophet")
st.divider()


@st.cache_data
def load_and_format_data(filename ='/home/shreya/mobius/time_series/streamlit_app/5-year Revenue by Month.xlsx', sheet_name = 'sales_formatted'):
    df = pd.read_excel(filename, sheet_name=sheet_name)
    df.rename(columns = {'Date':'ds', 'Sales':'y'}, inplace=True)
    train = df[:-12]
    test = df[-12:]

    return df,train, test

def init_model(changepoint_prior_scale, seasonality_prior_scale, add_holidays = True):
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale)
    if add_holidays:
        m.add_country_holidays(country_name = 'India')
    return m

def test_model(train, test, m):
    m.fit(train)
    future = m.make_future_dataframe(periods = 12, freq = 'MS')
    test_fcst = m.predict(future)
    # Plot the forecast with the actuals
    f, ax = plt.subplots(figsize=(15, 5))
    ax.plot(test['ds'], test['y'], color='r', marker = '.', label = 'actual values')
    fig = m.plot(test_fcst, ax=ax, include_legend=True)
    fig2 = plot_plotly(m, test_fcst)
    plt.legend()
    err = mean_absolute_percentage_error(y_true=test['y'], y_pred=test_fcst['yhat'][-12:]) * 100
    ax.set_title(f'Forecast with an error of {err:.2f}%')
    
    main_page.write(fig)
    main_page.write(fig2)

    st.session_state['model'] = m

def future_state():
    st.session_state['future'] = True

def download_state():
    if 'download' not in st.session_state:
        st.session_state['download'] = True
    st.session_state['download'] = True

def download_model():
    with open('serialized_model.json', 'w') as fout:
        fout.write(model_to_json(st.session_state['model']))  # Save model
    st.session_state['download'] = False

def forecast_future(df):
    model = init_model(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale, add_holidays=add_holidays)
    model.fit(df)
    future = model.make_future_dataframe(periods = 12, freq = 'MS')
    fcst = model.predict(future)
    fig = plot_plotly(model, fcst)
    st.write(model.changepoint_prior_scale)
    with placeholder.container():
        st.write(fig)    

    st.session_state['model'] = model
#hyperparameters
st.sidebar.title('Tune the model here:')
changepoint_prior_scale = st.sidebar.slider('Changepoint Prior Scale', help='It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.If it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. ',
                  min_value=0.001, max_value=0.5, value=0.05)

seasonality_prior_scale = st.sidebar.slider('Seasonality Prior Scale', help='This parameter controls the flexibility of the seasonality. A large value allows the seasonality to fit large fluctuations, a small value shrinks the magnitude of the seasonality.',
                                            min_value=0.01, max_value=10.0, value=10.0)
add_holidays = st.sidebar.toggle('Holidays', help='Models the effects of the major Indian holidays on the data')

st.sidebar.divider()

st.sidebar.title('Happy with the model?')
col1, col2 = st.sidebar.columns([0.4, 0.6])
col1.button('Forecast' , on_click=future_state, help='Predicts the data for the next 12 months based on complete history')
col2.button('Download Model', on_click=download_state, help='Saves the model to a serialised json file.')

placeholder = st.empty()
main_page = placeholder.container()


if 'future' not in st.session_state or st.session_state['future']==False:
    st.session_state['future'] = False
    load_state = main_page.text('Loading data...')

    df, train, test = load_and_format_data()
    if 'df' not in st.session_state:
        st.session_state['df'] = df
    load_state.text('Done!')

    m = init_model(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale, add_holidays=add_holidays)

    load_state.text('Model loaded!')

    test_model(train, test, m)

    load_state.subheader('Predicting for Test Data. Click on forecast to predict for the next 12 months.')

    if 'download' in st.session_state and st.session_state['download']==True:
        download_model()
    

elif st.session_state['future'] == True:
    placeholder.empty()
    forecast_future(st.session_state['df'])
    # st.session_state['future'] = False

    if 'download' in st.session_state and st.session_state['download']==True:
        download_model()
    

    

