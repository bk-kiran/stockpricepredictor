import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import cufflinks as cf
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt

st.set_page_config(page_title='Predicting Stock Prices', page_icon='📈')
st.title('Predicting Stock Prices 📈')
st.write("View and Predict Stock Data for the 5 FAAMG Companies!")

# changing between view and predict modes


def main():
    option_list = ['Select', 'View Previous Data', 'Predict Future Data']
    option = st.selectbox('Select an Option', option_list)
    if option == 'View Previous Data':
        view()
    elif option == 'Predict Future Data':
        predict()
    else:
        pass


@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df


# list of companies to choose from through a select box
ticker_list = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT']
option = st.selectbox('Select Company', ticker_list)
if option == 'AAPL':
    longname = 'Apple Inc.'
elif option == 'AMZN':
    longname = "Amazon.com, Inc."
elif option == 'GOOG':
    longname = "Alphabet Inc Class C (Google)"
elif option == 'META':
    longname = 'Meta Platforms Inc'
else:
    longname = 'Microsoft Corp'
option = option.upper()
today = datetime.date.today()
duration = 3000  # setting the duration to 3000 days of data from the current day
before = today - datetime.timedelta(days=duration)
start_date = before
end_date = today

data = download_data(option, start_date, end_date)
scaler = StandardScaler()


def view():
    st.header(f'Stock Data for {longname} for the last 30 days')
    # viewing the last 30 days of data on the dataframe
    st.dataframe(data.tail(30))
    st.header(f'Bollinger Bands for {longname}')  # inializing bollinger bands
    qf = cf.QuantFig(data, title='First Quant Figure',
                     legend='top', name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)


def predict():
    st.header(f'Predict Stock Data for {longname}')
    num = st.number_input(
        'How many days in the future would you like to predict?', value=1)  # finding the number of days to predict in the future
    num = int(num)
    if st.button('Predict'):
        engine = RandomForestRegressor()  # running a RandomForestRegressor engine
        model(engine, num)


def model(model, num):
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    # spliting the data into training/testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, random_state=7)

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    forecast_pred = model.predict(x_forecast)
    day = 1
    st.write(f"Close Price(s) for the next {num} day(s)")
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()
