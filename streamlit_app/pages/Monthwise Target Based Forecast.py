import streamlit as st

import pandas as pd
from prophet import Prophet


#read the data
df = pd.read_csv('/home/shreya/mobius/time_series/balance_sheet.csv')

st.title('Target Based Adjustement over Prophet Model Predictions')
st.subheader('For a given caption and target')


#dropdown menu has all the unique captions present in the df
st.sidebar.title('Select Caption')
caption = st.sidebar.selectbox('Select Caption', set(df['Display Caption'].values))


#filter based on caption
caption_df = df[df['Display Caption']==caption]
caption_df = caption_df.drop(['Display Caption'], axis = 1)

caption_df = caption_df.T
caption_df.columns = caption_df.iloc[1]
caption_df = caption_df.iloc[2:]


def make_predictions(caption_df):
    '''
    Function to make predictions for the next 12 months using prophet 
    '''
    outputs_df = pd.DataFrame()
    for line_item in caption_df.columns:
        m = Prophet()
        item_df = caption_df[line_item]
        item_df = pd.DataFrame(item_df)
        item_df.reset_index(inplace=True)
        item_df.columns = ['ds', 'y']
        m.fit(item_df)
        future = m.make_future_dataframe(periods = 12, freq = 'MS') #freq - month start
        fcst = m.predict(future)
        outputs_df[line_item] = fcst['yhat']

    return outputs_df, future

outputs_df, future = make_predictions(caption_df)
outputs_df.index = future['ds']

#to adjust the monthwise sum
month_wise = outputs_df.T
monthwise_sum = month_wise.sum() 
# cur_sum = monthwise_sum.sum() #total sum of dataframe
# already_achieved = cur_sum - monthwise_sum[-12:].sum() #achieved in the past
# cur_future_sum = cur_sum-already_achieved #forecasted achievement for the predicted period 
cur_future_sum = monthwise_sum[-12:].sum()

#show the original forecast
st.subheader(f"Original Forecast for {caption}")
st.write(month_wise)


def adjust_data(temp_df, month, new_month_target):
    month_sum = temp_df.sum().values[0]
    #caption_target = eval(input('Target for given month: '))
    new_fcst = pd.DataFrame(columns = [month])
    for i in temp_df.index:
        new_fcst.loc[i] =(temp_df.loc[i].values[0]/month_sum) * new_month_target
    return new_fcst

def adjust_month(month_wise, month, month_target):
    #month = st.sidebar.text_input('Month')
    temp_df = month_wise.loc[:,month] 
    temp_df = pd.DataFrame(temp_df)
    
    new_fcst = adjust_data(temp_df, month, month_target)
    return new_fcst



flag = 0

caption_target = st.sidebar.number_input('Target for the selected caption')

#wait for target input 
if caption_target:

    #list to store the new monthly targets
    new_monthwise_sum = []
    #rem_target = caption_target - already_achieved #total target for future 12 months 
    #rem_target = caption_target

    #split the rem target across the next 12 months
    for val in monthwise_sum[-12:]:
        new_val = (val/cur_future_sum * caption_target)
        new_monthwise_sum.append(new_val)
    
    new_df = pd.DataFrame()
    #for the next 12 months, adjust each line item based on the monthly target
    for idx, col in enumerate(month_wise.columns[-12:]):
        new_fcst = adjust_month(month_wise, col, new_monthwise_sum[idx])
        new_df[col] = new_fcst[col]

    #reset the future portion of the df
    month_wise.iloc[:, -12:] = new_df

    flag = 1

#display results 
if flag:
    st.subheader("Adjusted Forecast")
    #st.write(new_fcst)
    st.write(month_wise)

    #st.write(caption_target)
    st.write('Sum of caption: ', new_df.sum().sum())