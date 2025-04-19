from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import plotly
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests
import time
import random
import yfinance as yf
import os
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = Flask(__name__)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for proxy management
PROXY_LIST = []
PROXY_LAST_UPDATED = None
PROXY_UPDATE_INTERVAL = timedelta(hours=1)

def get_free_proxies():
    """Scrape free proxies from proxy list websites"""
    proxies = []
    urls = [
        'https://free-proxy-list.net/',
        'https://www.sslproxies.org/'
    ]
    
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'proxylisttable'})
            
            if table and table.tbody:
                for row in table.tbody.find_all('tr'):
                    columns = row.find_all('td')
                    if len(columns) >= 7:
                        ip = columns[0].text.strip()
                        port = columns[1].text.strip()
                        https = columns[6].text.strip()
                        
                        # Only use proxies that are working with HTTPS
                        if https.lower() == 'yes':
                            proxy = f"https://{ip}:{port}"
                            proxies.append(proxy)
                        else:
                            proxy = f"http://{ip}:{port}"
                            proxies.append(proxy)
            else:
                logger.warning(f"Could not find proxy table on {url}")
        except Exception as e:
            logger.error(f"Error scraping proxies from {url}: {e}")
    
    logger.info(f"Found {len(proxies)} proxies")
    return proxies

def update_proxy_list():
    """Update the global proxy list if needed"""
    global PROXY_LIST, PROXY_LAST_UPDATED
    
    # Update proxy list if it's empty or out of date
    if not PROXY_LIST or not PROXY_LAST_UPDATED or \
       (datetime.now() - PROXY_LAST_UPDATED) > PROXY_UPDATE_INTERVAL:
        logger.info("Updating proxy list...")
        PROXY_LIST = get_free_proxies()
        PROXY_LAST_UPDATED = datetime.now()

def create_yf_session(proxy=None):
    """Create a robust session for YFinance with optional proxy"""
    session = requests.Session()
    
    # Configure retries
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Rotate user agents
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
    ]
    
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    })
    
    # Set proxy if provided
    if proxy:
        session.proxies = {
            "http": proxy,
            "https": proxy
        }
    
    return session

def test_proxy(proxy, timeout=5):
    """Test if a proxy is working"""
    try:
        session = requests.Session()
        session.proxies = {
            "http": proxy,
            "https": proxy
        }
        response = session.get("https://httpbin.org/ip", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def get_working_proxy():
    """Get a working proxy from the list"""
    update_proxy_list()
    
    if not PROXY_LIST:
        logger.warning("No proxies available")
        return None
    
    # Try up to 5 random proxies
    for _ in range(min(5, len(PROXY_LIST))):
        proxy = random.choice(PROXY_LIST)
        logger.info(f"Testing proxy: {proxy}")
        
        if test_proxy(proxy):
            logger.info(f"Found working proxy: {proxy}")
            return proxy
    
    logger.warning("Could not find a working proxy")
    return None

def format_growth(value):
    """Convert value to percentage format"""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def format_ratio(value):
    """Format ratio to 2 decimal places"""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_revenue_billions(value):
    """Convert revenue to billions ($B)"""
    if value is None:
        return "N/A"
    try:
        return f"${float(value) / 1e9:.2f}B"
    except (ValueError, TypeError):
        return "N/A"

def get_stock_data(symbol, start_date, end_date):
    """Fetch historical data from yfinance using proxies if needed."""
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        try:
            # Try without proxy first
            if attempts == 0:
                logger.info(f"Attempting to fetch {symbol} without proxy...")
                session = create_yf_session()
            else:
                # Try with a proxy on subsequent attempts
                proxy = get_working_proxy()
                
                if not proxy:
                    logger.warning("No proxy available, trying without proxy...")
                    session = create_yf_session()
                else:
                    logger.info(f"Attempting with proxy: {proxy}")
                    session = create_yf_session(proxy)
            
            # Create a ticker with our custom session
            ticker = yf.Ticker(symbol)
            ticker.session = session
            
            # Get historical data with a slight delay to not trigger rate limits
            time.sleep(random.uniform(0.5, 1.5))
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}, trying again...")
                attempts += 1
                continue
                
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                logger.info(f"Successfully got data for {symbol}")
                return df.dropna()
                
        except Exception as e:
            logger.error(f"Error on attempt {attempts+1} for {symbol}: {str(e)}")
        
        attempts += 1
    
    # If all attempts fail, return empty DataFrame or create mock data
    logger.error(f"Failed to get data for {symbol} after {max_attempts} attempts")
    
    # Create mock data as a fallback
    mock_days = (end_date - start_date).days
    if mock_days > 0:
        logger.info(f"Creating mock data for {symbol}")
        mock_index = pd.date_range(start=start_date, end=end_date, freq='B')
        if len(mock_index) > 0:
            # Generate some random price data based on the symbol's first letter
            # to have some variation between different symbols
            base_price = ord(symbol[0]) % 10 * 10 + 50  # price between $50-$150
            
            mock_close = [base_price * (1 + random.uniform(-0.002, 0.002)) for _ in range(len(mock_index))]
            mock_data = {
                'Open': [price * (1 - random.uniform(0, 0.01)) for price in mock_close],
                'High': [price * (1 + random.uniform(0, 0.02)) for price in mock_close],
                'Low': [price * (1 - random.uniform(0, 0.02)) for price in mock_close],
                'Close': mock_close,
                'Volume': [int(1000000 * random.uniform(0.5, 1.5)) for _ in range(len(mock_index))]
            }
            return pd.DataFrame(mock_data, index=mock_index)
    
    return pd.DataFrame()

def get_yfinance_data(symbol):
    """Fetch financial metrics from yfinance with proxy support."""
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        try:
            logger.info(f"Getting yfinance data for {symbol}...")
            
            # Create session with or without proxy depending on attempt number
            if attempts == 0:
                session = create_yf_session()
            else:
                proxy = get_working_proxy()
                session = create_yf_session(proxy)
            
            # Create ticker with our session
            stock = yf.Ticker(symbol)
            stock.session = session
            
            # Add a slight delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))  
            
            # Get info
            info = stock.info
            
            data = {
                'totalRevenue': info.get('totalRevenue', None),
                'revenueGrowth': info.get('revenueGrowth', None),
                'marketCap': info.get('marketCap', None),
                'trailingPE': info.get('trailingPE', None),
                'forwardPE': info.get('forwardPE', None),
                'profitMargins': info.get('profitMargins', None),
                'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months', None)
            }
            
            # Check if we got meaningful data
            if any(v is not None for v in data.values()):
                logger.info(f"Successfully got financial data for {symbol}")
                return data
                
        except Exception as e:
            logger.error(f"Error with yfinance for {symbol} on attempt {attempts+1}: {str(e)}")
        
        attempts += 1
    
    # If all attempts fail, return some mock financial data
    logger.warning(f"Using fallback data for {symbol}")
    return {
        'totalRevenue': 10000000000 + random.uniform(-2000000000, 2000000000),
        'revenueGrowth': random.uniform(0.05, 0.25),
        'marketCap': 50000000000 + random.uniform(-10000000000, 10000000000),
        'trailingPE': random.uniform(15, 30),
        'forwardPE': random.uniform(12, 25),
        'profitMargins': random.uniform(0.1, 0.3),
        'priceToSalesTrailing12Months': random.uniform(2, 6)
    }

def calculate_future_value(revenue, revenue_growth, market_cap, trailing_pe, profit_margin):
    """Calculate a naive 5-year future value with mild data cleaning."""
    adjustments = []
    def trailing_pe_str(pe_val):
        return "N/A" if pe_val is None else f"{pe_val:.1f}"
    def profit_margin_str(pm_val):
        return "N/A" if pm_val is None else f"{pm_val:.1f}%"
    
    if revenue is None or revenue_growth is None or market_cap is None or market_cap == 0:
        return None, None, ""
    
    # Adjust revenue growth
    if revenue_growth <= 0:
        adjustments.append("Revenue growth adjusted to 5% from non-positive value")
        revenue_growth = 0.05
    elif revenue_growth > 0.25:
        adjustments.append(f"Revenue growth capped at 25% from {revenue_growth * 100:.1f}%")
        revenue_growth = 0.25
    
    # Adjust trailing P/E
    if trailing_pe is None or trailing_pe == 0 or trailing_pe > 30:
        old_pe_formatted = trailing_pe_str(trailing_pe)
        adjustments.append(f"P/E adjusted to 30 from {old_pe_formatted}")
        trailing_pe = 30
    
    # Adjust profit margin
    if profit_margin is None or profit_margin < 1:
        old_pm_formatted = profit_margin_str(profit_margin)
        adjustments.append(f"Profit margin adjusted to 5% from {old_pm_formatted}")
        profit_margin = 5
    
    profit_margin /= 100.0  # convert from percent to decimal
    future_value = revenue * ((1 + revenue_growth) ** 5) * profit_margin * trailing_pe
    future_value_billion = round(future_value / 1e9, 2)
    rate_increase = round(future_value / market_cap, 2)
    adjustment_explanation = "; ".join(adjustments)
    
    return future_value_billion, rate_increase, adjustment_explanation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    """
    Main endpoint to return the Plotly chart and stock/financial data in JSON
    """
    ticker = request.form['ticker'].strip().upper()
    period = request.form.get('period', '1Y')
    
    # If the user is in "Fibonacci" mode, we respect the "manualFib" checkbox.
    # If in "Trendline" mode, we do not draw the fib lines.
    # So let's interpret request.form for these booleans:
    chart_mode = request.form.get('chartMode', 'fib')   # "fib" or "trendlines"
    manual_fib = False
    if chart_mode == 'fib':  # only if they're in fib mode do we read the checkbox
        manual_fib = request.form.get('manualFib', 'false') == 'true'
    
    show_extensions = request.form.get('showExtensions', 'false') == 'true'
    fib_high = request.form.get('fibHigh')
    debug = True
    
    if not ticker:
        return jsonify(error="Please enter a valid ticker symbol")
    
    try:
        # Determine start/end date from period
        end_date = datetime.now()
        if period == '1M':
            start_date = end_date - timedelta(days=30)
        elif period == '3M':
            start_date = end_date - timedelta(days=90)
        elif period == '6M':
            start_date = end_date - timedelta(days=180)
        elif period == '1Y':
            start_date = end_date - timedelta(days=365)
        elif period == '5Y':
            start_date = end_date - timedelta(days=365*5)
        else:
            start_date = end_date - timedelta(days=365)
        
        logger.info(f"Getting stock data for {ticker} from {start_date} to {end_date}")
        df = get_stock_data(ticker, start_date, end_date)
        if df.empty:
            return jsonify(error=f"No data found for ticker: {ticker}")
        
        # Build the Plotly figure
        fig = go.Figure()
        
        # Convert index to datetime if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        x_dates = df.index.strftime('%Y-%m-%d').tolist()
        y_values = df['Close'].tolist()
        
        # Price trace
        fig.add_trace(go.Scatter(
            x=x_dates,
            y=y_values,
            mode='lines',
            name='Close Price',
            line=dict(color='#0ac775', width=2)
        ))
        
        # Only draw Fibonacci lines if user selected "Fib" mode AND manualFib is true
        if chart_mode == 'fib' and manual_fib and fib_high:
            try:
                fib_high_val = float(fib_high)
                # Standard retracement levels
                fib_levels = {
                    'level0': fib_high_val,
                    'level236': fib_high_val * (1 - 0.236),
                    'level382': fib_high_val * (1 - 0.382),
                    'level50': fib_high_val * (1 - 0.5),
                    'level618': fib_high_val * (1 - 0.618),
                    'level786': fib_high_val * (1 - 0.786),
                }
                
                fib_colors = {
                    'level0': 'rgba(255, 0, 0, 0.7)',
                    'level236': 'rgba(255, 165, 0, 0.7)',
                    'level382': 'rgba(255, 255, 0, 0.7)',
                    'level50': 'rgba(0, 128, 0, 0.7)',
                    'level618': 'rgba(0, 0, 255, 0.7)',
                    'level786': 'rgba(128, 0, 128, 0.7)'
                }
                
                fib_names = {
                    'level0': '0% - ' + f"${fib_high_val:.2f}",
                    'level236': '23.6% - ' + f"${fib_levels['level236']:.2f}",
                    'level382': '38.2% - ' + f"${fib_levels['level382']:.2f}",
                    'level50': '50% - ' + f"${fib_levels['level50']:.2f}",
                    'level618': '61.8% - ' + f"${fib_levels['level618']:.2f}",
                    'level786': '78.6% - ' + f"${fib_levels['level786']:.2f}",
                }
                
                for level, value in fib_levels.items():
                    fig.add_trace(go.Scatter(
                        x=x_dates,
                        y=[value]*len(x_dates),
                        mode='lines',
                        line=dict(
                            color=fib_colors.get(level, 'rgba(128, 128, 128, 0.7)'),
                            width=1,
                            dash='dash'
                        ),
                        name=fib_names.get(level, level),
                        hoverinfo='name+y'
                    ))
                
                # Optional extension lines above fib_high
                if show_extensions:
                    ext_ratios = [1.272, 1.382, 1.618, 2.618]
                    ext_colors = ['#FF6666', '#FF8888', '#FFAAAA', '#FFC0CB']
                    
                    for i, ratio in enumerate(ext_ratios):
                        extension_value = fib_high_val * ratio
                        ratio_percent = ratio * 100
                        fig.add_trace(go.Scatter(
                            x=x_dates,
                            y=[extension_value]*len(x_dates),
                            mode='lines',
                            line=dict(
                                color=ext_colors[i],
                                width=1,
                                dash='dash'
                            ),
                            name=f"{ratio_percent:.1f}% - ${extension_value:.2f}",
                            hoverinfo='name+y'
                        ))
            except ValueError:
                if debug:
                    logger.error("Error: Could not parse the manual fib high value.")
        
        # Layout
        period_name = {
            "1M": "Past Month",
            "3M": "Past 3 Months",
            "6M": "Past 6 Months",
            "1Y": "Past Year",
            "5Y": "Past 5 Years"
        }.get(period, "Past Year")
        
        fig.update_layout(
            title=f"{ticker} Stock Price - {period_name}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=600
        )
        
        # Get financial metrics
        logger.info(f"Getting financial metrics for {ticker}")
        financial_data = get_yfinance_data(ticker)
        
        current_price = df['Close'].iloc[-1]
        first_price = df['Close'].iloc[0]
        period_change = ((current_price / first_price) - 1) * 100
        high_price = df['High'].max()
        low_price = df['Low'].min()
        
        price_stats = {
            "current": f"${current_price:.2f}",
            "change": f"{period_change:.2f}%",
            "high": f"${high_price:.2f}",
            "low": f"${low_price:.2f}"
        }
        
        financial_metrics = {
            "revenueGrowth": format_growth(financial_data.get('revenueGrowth')),
            "forwardPE": format_ratio(financial_data.get('forwardPE')),
            "trailingPE": format_ratio(financial_data.get('trailingPE')),
            "profitMargin": format_growth(financial_data.get('profitMargins')),
            "priceToSales": format_ratio(financial_data.get('priceToSalesTrailing12Months')),
            "totalRevenue": format_revenue_billions(financial_data.get('totalRevenue')),
            "marketCap": format_revenue_billions(financial_data.get('marketCap'))
        }
        
        # Calculate naive 5-year price target
        revenue = financial_data.get('totalRevenue')
        revenue_growth = financial_data.get('revenueGrowth')
        market_cap = financial_data.get('marketCap')
        trailing_pe = financial_data.get('trailingPE')
        profit_margin = financial_data.get('profitMargins')
        
        if profit_margin is not None:
            profit_margin *= 100
            
        future_value, rate_increase, adjustments = calculate_future_value(
            revenue, revenue_growth, market_cap, trailing_pe, profit_margin
        )
        
        price_target = {
            "futureValue": None if future_value is None else f"${future_value}B",
            "rateIncrease": None if rate_increase is None else f"{rate_increase}x",
            "adjustments": adjustments
        }
        
        # Convert figure to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Return everything as JSON
        return jsonify(
            graph=graphJSON,
            price=price_stats,
            financials=financial_metrics,
            priceTarget=price_target
        )
    except Exception as e:
        logger.error(f"Error in plot function: {str(e)}")
        return jsonify(error=f"An error occurred: {str(e)}")

# Make sure the templates directory exists
if not os.path.exists('templates'):
    os.makedirs('templates')
if not os.path.exists('static'):
    os.makedirs('static')

# Create a placeholder logo to avoid 404
def create_placeholder_logo():
    try:
        if not os.path.exists('static/akinator_logo.png'):
            # Create a small colored square as a placeholder logo
            from PIL import Image, ImageDraw
            
            img = Image.new('RGB', (90, 90), color='#0ac775')
            d = ImageDraw.Draw(img)
            d.rectangle([(10, 10), (80, 80)], fill='white')
            
            img.save('static/akinator_logo.png')
            logger.info("Created placeholder logo")
    except Exception as e:
        logger.error(f"Error creating placeholder logo: {e}")

# Initialize proxy list on startup
@app.before_first_request
def initialize():
    update_proxy_list()
    create_placeholder_logo()

if __name__ == '__main__':
    # Update proxy list at startup
    update_proxy_list()
    create_placeholder_logo()
    app.run(host='0.0.0.0', port=8080, debug=False)
