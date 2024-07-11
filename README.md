<h1 style="text-align: center; color: #2E3B4E;">StockVision</h1>

<p style="font-size: 18px; line-height: 1.6;">StockVision is a web application built using Streamlit that provides tools for stock analysis and prediction using machine learning and financial data visualization techniques.</p>

<h2 style="color: #4CAF50;">Overview</h2>
<p style="font-size: 16px;">StockVision allows users to:</p>
<ul style="font-size: 16px;">
    <li>🧠 <b>Predict Stock Prices</b>: Use historical data and a trained machine learning model to forecast future prices.</li>
    <li>📊 <b>Interactive Charts</b>: Visualize historical closing prices and moving averages (100-day and 200-day).</li>
    <li>📈 <b>Financial Ratios</b>: Analyze key financial ratios such as P/E ratio, P/B ratio, and market capitalization.</li>
    <li>📉 <b>Candlestick Charts</b>: Generate candlestick charts for both historical and predicted stock prices.</li>
</ul>

<h2 style="color: #4CAF50;">Features</h2>
<ul style="font-size: 16px;">
    <li><b>Predictive Modeling</b>: Utilizes a machine learning model to forecast stock prices for the next 100 days based on historical trends.</li>
    <li><b>Interactive Charts</b>: Offers interactive plots for closing prices, moving averages, and financial ratios.</li>
    <li><b>Candlestick Charts</b>: Provides visualization of stock price movements using candlestick charts for historical and predicted data.</li>
    <li><b>Financial Analysis</b>: Displays key financial ratios to assist in fundamental analysis.</li>
</ul>

<h2 style="color: #4CAF50;">Getting Started</h2>
<p style="font-size: 16px;">To run StockVision locally:</p>
<ol style="font-size: 16px;">
    <li>Clone this repository:
        <pre><code>git clone &lt;repository_url&gt;
cd StockVision
</code></pre>
    </li>
    <li>Install the required Python packages:
        <pre><code>pip install -r requirements.txt
</code></pre>
    </li>
    <li>Run the Streamlit app:
        <pre><code>streamlit run app.py
</code></pre>
    </li>
    <li>Access the application in your web browser at <a href="http://localhost:8501">http://localhost:8501</a>.</li>
</ol>

<h2 style="color: #4CAF50;">Dependencies</h2>
<ul style="font-size: 16px;">
    <li>Streamlit</li>
    <li>Pandas</li>
    <li>Matplotlib</li>
    <li>NumPy</li>
    <li>yfinance</li>
    <li>Keras (for loading the machine learning model)</li>
    <li>scikit-learn (for data preprocessing and linear regression)</li>
    <li>mplfinance (for candlestick charts)</li>
</ul>

<h2 style="color: #4CAF50;">Usage</h2>
<ol style="font-size: 16px;">
    <li><b>Enter Stock Ticker</b>: Input the stock ticker symbol (e.g., INFY.BO) in the text box.</li>
    <li><b>Predictions</b>: View predicted stock prices for the next 100 days along with trend analysis.</li>
    <li><b>Time Series Analysis</b>: Explore historical closing prices and moving averages for different time periods (1 month, 3 months, 6 months, 1 year, 5 years, or full data).</li>
    <li><b>Financial Ratios</b>: Analyze key financial ratios like P/E ratio, P/B ratio, and market capitalization.</li>
</ol>

<h2 style="color: #4CAF50;">Examples</h2>
<ul style="font-size: 16px;">
    <li><b>Predictions vs Original</b>: Compare predicted stock prices with actual prices for validation.</li>
    <li><b>Trend Analysis</b>: Determine whether the stock is in an uptrend or downtrend based on predicted prices.</li>
    <li><b>Candlestick Charts</b>: Visualize stock price movements using candlestick charts for both historical and predicted data.</li>
</ul>
