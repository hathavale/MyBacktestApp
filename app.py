import io
import pandas as pd
import backtrader as bt
from flask import Flask, request, render_template
import numpy as np
from datetime import datetime

# Initialize the Flask application
app = Flask(__name__)

# --- Strategy Classes ---
class SMACrossover(bt.Strategy):
    params = (('fast_period', 10), ('slow_period', 30),)
    def __init__(self):
        self.fast_moving_average = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_moving_average = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_moving_average, self.slow_moving_average)
        self.order = None
        self.buy_price = None
        self.comm = None
        self.equity = []
        self.dates = []
        self.buy_signals = []
        self.sell_signals = []
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )
                self.buy_price = order.executed.price
                self.comm = order.executed.comm
                self.buy_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            elif order.issell():
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )
                self.sell_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    def next(self):
        self.equity.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0).isoformat())
        if self.order:
            return
        if not self.position:
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * 0.95) / price)
                self.order = self.buy(size=size)
        else:
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()

class EMACrossover(bt.Strategy):
    params = (('fast_period', 10), ('slow_period', 30),)
    def __init__(self):
        self.fast_ema = bt.indicators.EMA(self.data.close, period=self.p.fast_period)
        self.slow_ema = bt.indicators.EMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)
        self.order = None
        self.equity = []
        self.dates = []
        self.buy_signals = []
        self.sell_signals = []
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buy_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.sell_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    def next(self):
        self.equity.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0).isoformat())
        if self.order:
            return
        if not self.position:
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * 0.95) / price)
                self.order = self.buy(size=size)
        else:
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()

class RSIStrategy(bt.Strategy):
    params = (('rsi_period', 14), ('oversold', 30), ('overbought', 70),)
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.order = None
        self.equity = []
        self.dates = []
        self.buy_signals = []
        self.sell_signals = []
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buy_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.sell_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    def next(self):
        self.equity.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0).isoformat())
        if self.order:
            return
        if not self.position:
            if self.rsi < self.p.oversold:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}')
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * 0.95) / price)
                self.order = self.buy(size=size)
        else:
            if self.rsi > self.p.overbought:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}')
                self.order = self.sell()

class BollingerBandsStrategy(bt.Strategy):
    params = (('period', 20), ('devfactor', 2),)
    def __init__(self):
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.devfactor)
        self.order = None
        self.equity = []
        self.dates = []
        self.buy_signals = []
        self.sell_signals = []
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buy_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.sell_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    def next(self):
        self.equity.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0).isoformat())
        if self.order:
            return
        if not self.position:
            if self.data.close[0] > self.bbands.top[0]:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * 0.95) / price)
                self.order = self.buy(size=size)
        else:
            if self.data.close[0] < self.bbands.bot[0]:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()

class MACDStrategy(bt.Strategy):
    params = (('fast_period', 12), ('slow_period', 26), ('signal_period', 9),)
    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.p.fast_period, period_me2=self.p.slow_period, period_signal=self.p.signal_period)
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.order = None
        self.equity = []
        self.dates = []
        self.buy_signals = []
        self.sell_signals = []
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buy_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.sell_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    def next(self):
        self.equity.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0).isoformat())
        if self.order:
            return
        if not self.position:
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * 0.95) / price)
                self.order = self.buy(size=size)
        else:
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()

class BreakoutStrategy(bt.Strategy):
    params = (('lookback', 20),)
    def __init__(self):
        self.high = bt.indicators.Highest(self.data.high, period=self.p.lookback)
        self.low = bt.indicators.Lowest(self.data.low, period=self.p.lookback)
        self.order = None
        self.equity = []
        self.dates = []
        self.buy_signals = []
        self.sell_signals = []
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buy_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.sell_signals.append({'date': self.datas[0].datetime.date(0).isoformat(), 'price': self.data.close[0]})
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    def next(self):
        self.equity.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0).isoformat())
        if self.order:
            return
        if not self.position:
            if self.data.close[0] > self.high[-1]:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * 0.95) / price)
                self.order = self.buy(size=size)
        else:
            if self.data.close[0] < self.low[-1]:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    if 'data_file' not in request.files:
        return render_template('index.html', results={"message": "No file part in the request.", "error": True})

    file = request.files['data_file']
    if file.filename == '':
        return render_template('index.html', results={"message": "No selected file.", "error": True})

    if not file:
        return render_template('index.html', results={"message": "File upload failed.", "error": True})

    try:
        # Read CSV data
        data_io = io.StringIO(file.read().decode('utf-8'))
        df = pd.read_csv(data_io, parse_dates=['Date'], index_col='Date')

        # Validate and clean the DataFrame
        if df.index.isna().any():
            raise ValueError("CSV contains missing or invalid dates in the 'Date' column. Please ensure all dates are valid (e.g., YYYY-MM-DD).")
        df.index = pd.to_datetime(df.index, errors='coerce')
        if df.index.isna().any():
            raise ValueError("CSV contains unparseable dates in the 'Date' column. Please check the date format.")
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.sort_index()
        if df.empty:
            raise ValueError("CSV data is empty after removing invalid or missing entries.")

        # Capture start and end dates
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')

        # Get strategy and parameters
        strategy = request.form.get('strategy', 'sma_crossover')
        risk_free_rate = float(request.form.get('risk_free_rate', 3.0)) / 100

        # Validate risk-free rate
        if risk_free_rate < 0:
            raise ValueError("Risk-free rate must be non-negative.")

        # Initialize Cerebro
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days)
        cerebro.broker.setcommission(commission=0.0005)  # 0.05%

        # Select strategy and parameters
        strategy_name = {
            'sma_crossover': 'SMA Crossover',
            'ema_crossover': 'EMA Crossover',
            'rsi': 'RSI Mean Reversion',
            'bollinger_bands': 'Bollinger Bands Breakout',
            'macd': 'MACD Crossover',
            'breakout': 'Price Channel Breakout'
        }.get(strategy, 'SMA Crossover')

        params = {}
        if strategy == 'sma_crossover':
            params['sma_fast_period'] = int(request.form.get('sma_fast_period', 10))
            params['sma_slow_period'] = int(request.form.get('sma_slow_period', 30))
            if params['sma_fast_period'] <= 0 or params['sma_slow_period'] <= 0:
                raise ValueError("SMA periods must be positive.")
            cerebro.addstrategy(SMACrossover, fast_period=params['sma_fast_period'], slow_period=params['sma_slow_period'])
        elif strategy == 'ema_crossover':
            params['ema_fast_period'] = int(request.form.get('ema_fast_period', 10))
            params['ema_slow_period'] = int(request.form.get('ema_slow_period', 30))
            if params['ema_fast_period'] <= 0 or params['ema_slow_period'] <= 0:
                raise ValueError("EMA periods must be positive.")
            cerebro.addstrategy(EMACrossover, fast_period=params['ema_fast_period'], slow_period=params['ema_slow_period'])
        elif strategy == 'rsi':
            params['rsi_period'] = int(request.form.get('rsi_period', 14))
            params['oversold'] = float(request.form.get('oversold', 30))
            params['overbought'] = float(request.form.get('overbought', 70))
            if params['rsi_period'] <= 0:
                raise ValueError("RSI period must be positive.")
            if params['oversold'] < 0 or params['oversold'] > 100 or params['overbought'] < 0 or params['overbought'] > 100:
                raise ValueError("RSI thresholds must be between 0 and 100.")
            cerebro.addstrategy(RSIStrategy, rsi_period=params['rsi_period'], oversold=params['oversold'], overbought=params['overbought'])
        elif strategy == 'bollinger_bands':
            params['bb_period'] = int(request.form.get('bb_period', 20))
            params['devfactor'] = float(request.form.get('devfactor', 2.0))
            if params['bb_period'] <= 0:
                raise ValueError("Bollinger Bands period must be positive.")
            if params['devfactor'] <= 0:
                raise ValueError("Standard deviation multiplier must be positive.")
            cerebro.addstrategy(BollingerBandsStrategy, period=params['bb_period'], devfactor=params['devfactor'])
        elif strategy == 'macd':
            params['macd_fast_period'] = int(request.form.get('macd_fast_period', 12))
            params['macd_slow_period'] = int(request.form.get('macd_slow_period', 26))
            params['macd_signal_period'] = int(request.form.get('macd_signal_period', 9))
            if params['macd_fast_period'] <= 0 or params['macd_slow_period'] <= 0 or params['macd_signal_period'] <= 0:
                raise ValueError("MACD periods must be positive.")
            cerebro.addstrategy(MACDStrategy, fast_period=params['macd_fast_period'], slow_period=params['macd_slow_period'], signal_period=params['macd_signal_period'])
        elif strategy == 'breakout':
            params['lookback'] = int(request.form.get('lookback', 20))
            if params['lookback'] <= 0:
                raise ValueError("Breakout lookback period must be positive.")
            cerebro.addstrategy(BreakoutStrategy, lookback=params['lookback'])
        else:
            raise ValueError("Invalid strategy selected.")

        # Data feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest='OpenInterest' if 'OpenInterest' in df.columns else None
        )
        cerebro.adddata(data)

        # Run backtest
        initial_portfolio_value = cerebro.broker.getvalue()
        print("Running backtest...")
        strategies = cerebro.run()
        print("Backtest finished.")
        final_portfolio_value = cerebro.broker.getvalue()
        profit = final_portfolio_value - initial_portfolio_value
        profit_percent = (profit / initial_portfolio_value) * 100 if initial_portfolio_value else 0

        # Calculate Sharpe Ratio
        returns = strategies[0].analyzers.timereturn.get_analysis()
        if returns:
            daily_returns = pd.Series(returns)
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            annual_return = mean_daily_return * 252
            annual_std = std_daily_return * np.sqrt(252)
            sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std != 0 else 0
        else:
            sharpe_ratio = 0

        # Prepare chart data
        chart_data = {
            'labels': strategies[0].dates,
            'equity': strategies[0].equity,
            'buy_signals': [{'x': s['date'], 'y': s['price']} for s in strategies[0].buy_signals],
            'sell_signals': [{'x': s['date'], 'y': s['price']} for s in strategies[0].sell_signals]
        }

        # Prepare results
        results = {
            'initial_value': f"${initial_portfolio_value:,.2f}",
            'final_value': f"${final_portfolio_value:,.2f}",
            'profit': f"${profit:,.2f}",
            'profit_percent': f"{profit_percent:,.2f}%",
            'sma_fast_period': params.get('sma_fast_period'),
            'sma_slow_period': params.get('sma_slow_period'),
            'ema_fast_period': params.get('ema_fast_period'),
            'ema_slow_period': params.get('ema_slow_period'),
            'rsi_period': params.get('rsi_period'),
            'oversold': params.get('oversold'),
            'overbought': params.get('overbought'),
            'bb_period': params.get('bb_period'),
            'devfactor': params.get('devfactor'),
            'macd_fast_period': params.get('macd_fast_period'),
            'macd_slow_period': params.get('macd_slow_period'),
            'macd_signal_period': params.get('macd_signal_period'),
            'lookback': params.get('lookback'),
            'sharpe_ratio': f"{sharpe_ratio:.2f}",
            'risk_free_rate': f"{risk_free_rate * 100:.2f}%",
            'start_date': start_date,
            'end_date': end_date,
            'chart_data': chart_data,
            'strategy': strategy,
            'strategy_name': strategy_name,
            'message': "Backtest completed successfully!",
            'error': False
        }

    except ValueError as ve:
        results = {
            'message': f"Data error: {str(ve)}",
            'error': True
        }
    except Exception as e:
        results = {
            'message': f"An error occurred during backtesting: {str(e)}",
            'error': True
        }

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)