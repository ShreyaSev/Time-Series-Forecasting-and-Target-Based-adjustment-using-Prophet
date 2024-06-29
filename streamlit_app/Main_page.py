import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Prophet 👋")

st.sidebar.success("Select an app above.")

st.markdown(
    """
    Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

    We use a decomposable time series model (Harvey & Peters 1990) with three main
    model components: trend, seasonality, and holidays. They are combined in the following
    equation:
"""

)

st.image('streamlit_app/prophet_equation.png')

st.markdown("""
            Here g(t) is the trend function which models non-periodic changes in the value of the
time series, s(t) represents periodic changes (e.g., weekly and yearly seasonality), and
h(t) represents the effects of holidays which occur on potentially irregular schedules over
one or more days. The error term t represents any idiosyncratic changes which are not
accommodated by the model
            """
            )