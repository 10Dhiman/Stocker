from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import yfinance as yf
import pandas_ta as ta
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from bs4 import BeautifulSoup
import os
import sqlite3
import numpy as np
import requests
import io
import base64


app = Flask(__name__, template_folder='./templates', static_folder='./static')
secret_key = os.urandom(24)
app.secret_key = secret_key
DATABASE = 'community_posts.db'
# Initialize MLPClassifier outside of the route function
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=1000)

def init_db():
    with app.app_context():
        db = get_db()
        with open('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def generate_plot(df, plot_path):
    fig, ax = plt.subplots(figsize=(8, 4))  
    df.plot(y='Close', ax=ax)
    plt.savefig(plot_path)
    plt.close()

def get_start_date(data_range):
    today = date.today()
    if data_range == '0.5_year':
        return today - timedelta(days=180), today, '6 Months'
    elif data_range == '1_year':
        return today - timedelta(days=365), today, '1 Year'
    elif data_range == '2_years':
        return today - timedelta(days=2*365), today, '2 Years'
    elif data_range == '3_years':
        return today - timedelta(days=3*365), today, '3 Years'
    elif data_range == '4_years':
        return today - timedelta(days=4*365), today, '4 Years'
    elif data_range == '5_years':
        return today - timedelta(days=5*365), today, '5 Years'
    elif data_range == '10_years':
        return today - timedelta(days=10*365), today, '10 Years'
    else:
        return today - timedelta(days=365), today, '1 Year'  # Default to 1 year if invalid option selected

@app.route('/help.html')
def help():
    db = get_db()
    posts = db.execute('SELECT * FROM posts ORDER BY created_at DESC').fetchall()
    return render_template('help.html', posts=posts)

@app.route('/add_post', methods=['POST'])
def add_post():
    question = request.form['question']
    if question:
        db = get_db()
        db.execute('INSERT INTO posts (content) VALUES (?)', (question,))
        db.commit()
    return redirect(url_for('help'))

@app.route('/slide.html')
def home():
    return render_template('slide.html')

@app.route('/', methods=['GET'])
def slide():
    return render_template('slide.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    train_accuracy = test_accuracy = plot_url = prediction = None
    train_precision = test_precision = train_recall = test_recall = train_f1 = test_f1 = None
    company_name = time_period = error_message = None
    open_price = close_price = profit_or_loss = None  # Initialize profit_or_loss here

    if request.method == 'POST':
        company_name = request.form['company_name']
        data_range = request.form['data_range']
        
        if company_name:
            start_date, end_date, time_period = get_start_date(data_range)
            df = yf.download(company_name, start=start_date, end=end_date, progress=False)

            if len(df) < 20:
                error_message = 'Not enough data available for training and testing.'
                return render_template('index.html', error_message=error_message)

            df['RSI(2)'] = ta.rsi(df['Close'], length=2)
            df['RSI(7)'] = ta.rsi(df['Close'], length=7)
            df['RSI(14)'] = ta.rsi(df['Close'], length=14)
            df['CCI(30)'] = ta.cci(close=df['Close'], length=30, high=df['High'], low=df['Low'])
            df['CCI(50)'] = ta.cci(close=df['Close'], length=50, high=df['High'], low=df['Low'])
            df['CCI(100)'] = ta.cci(close=df['Close'], length=100, high=df['High'], low=df['Low'])

            df['LABEL'] = np.where(df['Open'].shift(-1).gt(df['Open']), "UP", "DOWN")
            df = df.dropna()

            if not df.empty:
                X = df[df.columns[6:-1]].values
                y = df['LABEL'].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                mlp.fit(X_train, y_train)

                predict_train = mlp.predict(X_train)
                predict_test = mlp.predict(X_test)

                train_accuracy = accuracy_score(y_train, predict_train)
                test_accuracy = accuracy_score(y_test, predict_test)
                train_precision = precision_score(y_train, predict_train, average='weighted')
                test_precision = precision_score(y_test, predict_test, average='weighted')
                train_recall = recall_score(y_train, predict_train, average='weighted')
                test_recall = recall_score(y_test, predict_test, average='weighted')
                train_f1 = f1_score(y_train, predict_train, average='weighted')
                test_f1 = f1_score(y_test, predict_test, average='weighted')

                if not os.path.exists('static'):
                    os.makedirs('static')

                plot_path = f'static/plot_{company_name}.png'
                generate_plot(df, plot_path)

                df['Prediction'] = mlp.predict(X)
                tomorrow_prediction = mlp.predict(X[-1].reshape(1, -1))

                plot_url = plot_path
                prediction = tomorrow_prediction[0]

                # Get open and close prices
                open_price = df.iloc[-1]['Open']
                close_price = df.iloc[-1]['Close']

                # Calculate profit or loss
                profit_or_loss = "Profitable" if close_price > open_price else "Risky"
                if abs(close_price - open_price) > 0.1 * open_price:
                    profit_or_loss = "High Risk"

    return render_template('index.html', company_name=company_name, time_period=time_period,
                           train_accuracy=train_accuracy, test_accuracy=test_accuracy,
                           train_precision=train_precision, test_precision=test_precision,
                           train_recall=train_recall, test_recall=test_recall,
                           train_f1=train_f1, test_f1=test_f1,
                           plot_url=plot_url, prediction=prediction, error_message=error_message,
                           open_price=open_price, close_price=close_price, profit_or_loss=profit_or_loss)

@app.route('/prediction.html')
def predictions():
    url = "https://finance.yahoo.com/trending-tickers"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract stock data
    rows = soup.find_all('tr', {'class': 'simpTblRow'})
    stocks = []

    for row in rows[:5]:  # Get top 5 trending stocks
        symbol = row.find('td', {'aria-label': 'Symbol'}).text
        last_price = row.find('td', {'aria-label': 'Last Price'}).text
        change = row.find('td', {'aria-label': 'Change'}).text
        percent_change = row.find('td', {'aria-label': '% Change'}).text
        volume = row.find('td', {'aria-label': 'Volume'}).text
        market_cap = row.find('td', {'aria-label': 'Market Cap'}).text

        stocks.append({
            'symbol': symbol,
            'last_price': last_price,
            'change': change,
            'percent_change': percent_change,
            'volume': volume,
            'market_cap': market_cap
        })

    return render_template('prediction.html', stocks=stocks)

#premium 

def fetch_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data.reset_index(inplace=True)  # Ensure 'Date' column exists
    stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')  # Format the Date column
    return stock_data

def calculate_financial_ratios(stock_data):
    stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    return stock_data


def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {
        'Market Cap': info.get('marketCap'),
        'P/E Ratio': info.get('trailingPE'),
        'EPS': info.get('trailingEps'),
        'Dividend Yield': info.get('dividendYield'),
        'P/B Ratio': info.get('priceToBook'),
        'ROE': info.get('returnOnEquity'),
        'Debt to Equity': info.get('debtToEquity'),
        'Revenue': info.get('totalRevenue'),
        'Net Income': info.get('netIncomeToCommon'),
        'EBITDA': info.get('ebitda'),
        'Free Cash Flow': info.get('freeCashflow')
    }
    return fundamentals

def calculate_macd(stock_data):
    stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
    stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    return stock_data

def calculate_bollinger_bands(stock_data):
    stock_data['20 Day MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['20 Day STD'] = stock_data['Close'].rolling(window=20).std()
    stock_data['Upper Band'] = stock_data['20 Day MA'] + (stock_data['20 Day STD'] * 2)
    stock_data['Lower Band'] = stock_data['20 Day MA'] - (stock_data['20 Day STD'] * 2)
    return stock_data

def calculate_stochastic_oscillator(stock_data):
    stock_data['14-High'] = stock_data['High'].rolling(window=14).max()
    stock_data['14-Low'] = stock_data['Low'].rolling(window=14).min()
    stock_data['%K'] = (stock_data['Close'] - stock_data['14-Low']) * 100 / (stock_data['14-High'] - stock_data['14-Low'])
    stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()
    return stock_data

def predict_future_prices(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).map(pd.Timestamp.timestamp)
    X = stock_data[['Date']]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def generate_graphs(stock_data, model, ticker):
    fig, axes = plt.subplots(3, 1, figsize=(8, 4))

    # Stock price and moving averages
    stock_data[['Date', 'Close', 'SMA50', 'SMA200']].set_index('Date').plot(ax=axes[0])
    axes[0].set_title(f'{ticker} Stock Price and Moving Averages')
    axes[0].set_ylabel('Price')

    # RSI
    stock_data[['Date', 'RSI']].set_index('Date').plot(ax=axes[1])
    axes[1].set_title(f'{ticker} RSI')
    axes[1].set_ylabel('RSI')
    axes[1].axhline(70, color='r', linestyle='--')
    axes[1].axhline(30, color='g', linestyle='--')

    # Stock price prediction
    future_dates = [stock_data['Date'].iloc[-1] + 86400]  # Predicting the next day
    future_dates = pd.DataFrame(future_dates, columns=['Date'])
    future_price = model.predict(future_dates)
    axes[2].plot(stock_data['Date'], stock_data['Close'], label='Actual')
    axes[2].plot(future_dates['Date'], future_price, 'ro', label='Predicted')
    axes[2].set_title(f'{ticker} Stock Price Prediction')
    axes[2].set_ylabel('Price')
    axes[2].legend()

    # Save the graph
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return graph_url

def analyze_sentiment():
    # Placeholder for sentiment analysis
    return "Market sentiment is neutral (0)."

def analyze_trend(stock_data):
    sma50 = stock_data['SMA50'].iloc[-1]
    sma200 = stock_data['SMA200'].iloc[-1]
    rsi = stock_data['RSI'].iloc[-1]
    trend = "The stock is not in an uptrend." if sma50 <= sma200 else "The stock is in an uptrend."
    rsi_analysis = f"RSI ({rsi}) indicates the stock is {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neither overbought nor oversold'}."
    return trend, rsi_analysis

def analyze_technical(stock_data):
    macd = stock_data['MACD'].iloc[-1]
    signal_line = stock_data['Signal Line'].iloc[-1]
    upper_band = stock_data['Upper Band'].iloc[-1]
    lower_band = stock_data['Lower Band'].iloc[-1]
    bollinger_within_range = "Price is within the Bollinger Bands range."
    stochastic_osc = stock_data['%K'].iloc[-1]
    stochastic_analysis = f"Stochastic Oscillator ({stochastic_osc}) indicates {'overbought' if stochastic_osc > 80 else 'oversold' if stochastic_osc < 20 else 'neutral'} conditions."
    macd_analysis = "MACD is bearish." if macd < signal_line else "MACD is bullish."
    return macd_analysis, bollinger_within_range, stochastic_analysis

@app.route('/premium.html', methods=['GET', 'POST'])
def premium():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            return render_template('premium.html', error="No data available for the specified date range.")
        stock_data = calculate_financial_ratios(stock_data)
        stock_data = calculate_macd(stock_data)
        stock_data = calculate_bollinger_bands(stock_data)
        stock_data = calculate_stochastic_oscillator(stock_data)
        fundamentals = fetch_fundamental_data(ticker)
        model = predict_future_prices(stock_data)
        graph_url = generate_graphs(stock_data, model, ticker)

        sentiment = analyze_sentiment()
        trend, rsi_analysis = analyze_trend(stock_data)
        macd_analysis, bollinger_within_range, stochastic_analysis = analyze_technical(stock_data)
        future_price = model.predict(pd.DataFrame([stock_data['Date'].iloc[-1] + 86400], columns=['Date']))[0]
        investment_decision = "Investment Decision: Based on the analysis, it is recommended to make an informed decision."

        return render_template('premium.html', 
                               stock_data=stock_data.to_html(), 
                               fundamentals=fundamentals, 
                               graph_url=graph_url, 
                               sentiment=sentiment, 
                               trend=trend, 
                               rsi_analysis=rsi_analysis, 
                               macd_analysis=macd_analysis, 
                               bollinger_within_range=bollinger_within_range, 
                               stochastic_analysis=stochastic_analysis, 
                               future_price=future_price, 
                               investment_decision=investment_decision)
    return render_template('premium.html')
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'Stocker' and password == '161002':
            session['logged_in'] = True
            return redirect(url_for('premium'))  # Redirect to the premium route
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)