from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.linear_model import LinearRegression
import mplfinance as mpf
from stocknews import StockNews
# from newsapi import NewsApiClient

# # Initialize the News API client
# newsapi = NewsApiClient(api_key='1ed9b8797ae64f18afd07d6a4e331ad1')
start = '2013-07-11'
end = '2024-07-18'
st.markdown("<h1 style='text-align: center; font-size: 36px;'>StockVision</h1>", unsafe_allow_html=True)
st.title('Stock Trend Prediction for BSE')

tickers = []
user_input = st.text_input('Enter Stock Ticker', 'INFY.BO')
tickers.append(user_input)

data = {}
ratios = []
for ticker in tickers:
    stock_data = yf.download(ticker, start, end)
    stock_data['Ticker'] = ticker
    data[ticker] = stock_data


    ticker_info = yf.Ticker(ticker).info
    pe_ratio = ticker_info.get('trailingPE', 'N/A')
    pb_ratio = ticker_info.get('priceToBook', 'N/A')
    Mkt_Cap = ticker_info.get('marketCap','N/A')
    # Divid_rate =ticker_info.get('dividendRate','N/A')
    # Volume = ticker_info.get('volume','N/A')
    # wr = ticker_info.get('52 Week Range','N/A')
    ratios.append({
        'Ticker': ticker,
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'Mkt_Cap':Mkt_Cap,
        # 'Dividend_rate':Divid_rate,
        # 'Volume':Volume,
        # '52 Week Range':wr
    })

# Concatenate stock data
df = pd.concat(data.values())
df.reset_index(inplace=True)

# Create a DataFrame for financial ratios
ratios_df = pd.DataFrame(ratios)

# Display data and ratios
st.subheader('Data from 2013-2024')
st.write(df.describe())

st.subheader('Financial Ratios')
st.write(ratios_df)

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma200)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'g')
plt.plot(ma200,'r')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(data_training)
test_scaled = scaler.fit_transform(data_testing)

X_train = []
y_train = []
for i in range(100,train_scaled.shape[0]):
    X_train.append(train_scaled[i-100:i])
    y_train.append(train_scaled[i,0])

x_train , y_train  = np.array(X_train),np.array(y_train)

model = load_model('model.h5')

train_scaled_df = pd.DataFrame(train_scaled, columns=data_training.columns)
test_scaled_df = pd.DataFrame(test_scaled, columns=data_testing.columns)
past_100_days= data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test  = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_predicted =y_predicted*scale_factor
y_test = y_test*scale_factor

st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

last_100_days = final_df[-100:]
next_10_days = []
input_seq = scaler.transform(last_100_days)

for _ in range(10):
    input_seq_reshaped = input_seq[-100:].reshape(1, -1, 1)
    predicted_price = model.predict(input_seq_reshaped)
    next_10_days.append(predicted_price[0, 0])
    input_seq = np.append(input_seq, predicted_price, axis=0)

next_10_days = np.array(next_10_days).reshape(-1, 1)
next_10_days = scaler.inverse_transform(next_10_days)

st.subheader('Next 10 Days Prediction')
fig3 = plt.figure(figsize=(12,6))
plt.plot(next_10_days, 'g', label='Predicted Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

st.subheader('Trend Analysis for Next 10 Predicted Days')
X_next_10 = np.arange(len(next_10_days)).reshape(-1, 1)
y_next_10 = next_10_days.flatten()

regressor_next_10 = LinearRegression()
regressor_next_10.fit(X_next_10, y_next_10)
slope_next_10 = regressor_next_10.coef_[0]

if slope_next_10 > 0:
    trend_next_10 = 'Uptrend'
else:
    trend_next_10 = 'Downtrend'

st.write(f"The trend for the next 100 predicted days is: {trend_next_10}")

# Plot the trend line for the next 100 predicted days
fig4 = plt.figure(figsize=(12, 6))
plt.plot(X_next_10, y_next_10, label='Predicted Prices')
plt.plot(X_next_10, regressor_next_10.predict(X_next_10), label=f'Trend Line ({trend_next_10})', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)


st.subheader('Candlestick Chart for Last 100 Days')
last_100_days_candle = df.iloc[-100:, :]
mpf_data = last_100_days_candle[['Date', 'Open', 'High', 'Low', 'Close']]
mpf_data.set_index('Date', inplace=True)
mpf_fig, mpf_ax = mpf.plot(mpf_data, type='candle', style='charles', returnfig=True)
st.pyplot(mpf_fig)

# Candlestick chart for the next 10 predicted days
st.subheader('Candlestick Chart for Next 10 Predicted Days')
# Assuming you have open, high, low prices for the predicted days, here we generate dummy data for them
predicted_df = pd.DataFrame({
    'Date': pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10, freq='B'),
    'Open': y_next_10 * (1 + np.random.normal(0, 0.01, size=10)),  # Adding small noise
    'High': y_next_10 * (1 + np.random.uniform(0.01, 0.02, size=10)),  # Adding small noise
    'Low': y_next_10 * (1 - np.random.uniform(0.01, 0.02, size=10)),  # Adding small noise
    'Close': y_next_10
})

predicted_df.set_index('Date', inplace=True)
mpf_fig_pred, mpf_ax_pred = mpf.plot(predicted_df, type='candle', style='charles', returnfig=True)
st.pyplot(mpf_fig_pred)

if 'show_last_1_month' not in st.session_state:
    st.session_state.show_last_1_month = False
if 'show_last_3_months' not in st.session_state:
    st.session_state.show_last_3_months = False
if 'show_last_6_months' not in st.session_state:
    st.session_state.show_last_6_months = False
if 'show_last_1_year' not in st.session_state:
    st.session_state.show_last_1_year = False
if 'show_last_5_years' not in st.session_state:
    st.session_state.show_last_5_years = False
if 'show_full_data' not in st.session_state:
    st.session_state.show_full_data = False

# Function to plot closing prices for a given time period
def plot_closing_prices(data, title):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    st.pyplot(fig)

# Toggle button states and plots
if st.button('Last 1 Month'):
    st.session_state.show_last_1_month = not st.session_state.show_last_1_month
    st.session_state.show_last_3_months = False
    st.session_state.show_last_6_months = False
    st.session_state.show_last_1_year = False
    st.session_state.show_last_5_years = False
    st.session_state.show_full_data = False

if st.button('Last 3 Months'):
    st.session_state.show_last_3_months = not st.session_state.show_last_3_months
    st.session_state.show_last_1_month = False
    st.session_state.show_last_6_months = False
    st.session_state.show_last_1_year = False
    st.session_state.show_last_5_years = False
    st.session_state.show_full_data = False

if st.button('Last 6 Months'):
    st.session_state.show_last_6_months = not st.session_state.show_last_6_months
    st.session_state.show_last_1_month = False
    st.session_state.show_last_3_months = False
    st.session_state.show_last_1_year = False
    st.session_state.show_last_5_years = False
    st.session_state.show_full_data = False

if st.button('Last 1 Year'):
    st.session_state.show_last_1_year = not st.session_state.show_last_1_year
    st.session_state.show_last_1_month = False
    st.session_state.show_last_3_months = False
    st.session_state.show_last_6_months = False
    st.session_state.show_last_5_years = False
    st.session_state.show_full_data = False

if st.button('Last 5 Years'):
    st.session_state.show_last_5_years = not st.session_state.show_last_5_years
    st.session_state.show_last_1_month = False
    st.session_state.show_last_3_months = False
    st.session_state.show_last_6_months = False
    st.session_state.show_last_1_year = False
    st.session_state.show_full_data = False

if st.button('Full Data'):
    st.session_state.show_full_data = not st.session_state.show_full_data
    st.session_state.show_last_1_month = False
    st.session_state.show_last_3_months = False
    st.session_state.show_last_6_months = False
    st.session_state.show_last_1_year = False
    st.session_state.show_last_5_years = False

# Display plots based on the button states
if st.session_state.show_last_1_month:
    last_1_month = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(months=1))]
    plot_closing_prices(last_1_month, 'Closing Prices - Last 1 Month')

if st.session_state.show_last_3_months:
    last_3_months = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(months=3))]
    plot_closing_prices(last_3_months, 'Closing Prices - Last 3 Months')

if st.session_state.show_last_6_months:
    last_6_months = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(months=6))]
    plot_closing_prices(last_6_months, 'Closing Prices - Last 6 Months')

if st.session_state.show_last_1_year:
    last_1_year = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(years=1))]
    plot_closing_prices(last_1_year, 'Closing Prices - Last 1 Year')

if st.session_state.show_last_5_years:
    last_5_years = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(years=5))]
    plot_closing_prices(last_5_years, 'Closing Prices - Last 5 Years')

if st.session_state.show_full_data:
    plot_closing_prices(df, 'Closing Prices - Full Data')

# st.subheader(f'Latest News of {user_input}')
# sn = StockNews(user_input,save_news=False)
# df_news = sn.read_rss()
# for i in range(10):
#     st.subheader(f'News {i+1}')
#     st.write(df_news['published'][i])
#     st.write(df_news['title'][i])
#     st.write(df_news['summary'][i])
#     title_sentiment = df_news['sentiment_title'][i]
#     st.write(f'Title Sentiment {title_sentiment}')
#     news_sentiment = df_news['sentiment_summary'][i]
#     st.write(f'News Sentiment {news_sentiment}')

# st.markdown(f"""
# <h1 style="text-align: center; color: #2E3B4E;">Latest News of {user_input}</h1>
# """, unsafe_allow_html=True)

# sn = StockNews(user_input, save_news=False)
# df_news = sn.read_rss()

# for i in range(10):
#     st.markdown(f"""
#     <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
#         <h3 style="color: #2E3B4E;">News {i+1}</h3>
#         <p style="color: #6c757d;">{df_news['published'][i]}</p>
#         <h4 style="color: #007bff;"><a href="{df_news['link'][i]}" target="_blank" style="text-decoration: none; color: #007bff;">{df_news['title'][i]}</a></h4>
#         <p style="color: #343a40;">{df_news['summary'][i]}</p>
#         <p><b>Title Sentiment:</b> {df_news['sentiment_title'][i]}</p>
#         <p><b>News Sentiment:</b> {df_news['sentiment_summary'][i]}</p>
#     </div>
#     """, unsafe_allow_html=True)

st.markdown(f"""
<h1 style="text-align: center; color: #2E3B4E;">Latest News of {user_input}</h1>
""", unsafe_allow_html=True)

sn = StockNews(user_input, save_news=False)
df_news = sn.read_rss()
for i in range(10):
        st.markdown(f"""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <h3 style="color: #2E3B4E;">News {i+1}</h3>
            <p style="color: #6c757d;">{df_news['published'][i]}</p>
            <h4 style="color: #007bff;">{df_news['title'][i]}</h4>
            <p style="color: #343a40;">{df_news['summary'][i]}</p>
            <p style="color:#2E3B4E;"><b>Title Sentiment:</b> {df_news['sentiment_title'][i]}</p>
            <p style="color:#2E3B4E;"><b>News Sentiment:</b> {df_news['sentiment_summary'][i]}</p>
        </div>
        """, unsafe_allow_html=True)


