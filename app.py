import io
import pandas as pd
import backtrader as bt
from flask import Flask, request, render_template

# Initialize the Flask application
app = Flask(__name__)

# --- Backtrader Strategy Definition ---
# This is a simple SMA (Simple Moving Average) Crossover strategy.
# It buys when the fast moving average crosses above the slow moving average,
# and sells when the fast moving average crosses below the fast moving average.
class SMACrossover(bt.Strategy):
    # Define parameters for the strategy with default values
    params = (('fast_period', 10), ('slow_period', 30),)

    def __init__(self):
        # Create Simple Moving Average indicators
        self.fast_moving_average = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_moving_average = bt.indicators.SMA(self.data.close, period=self.p.slow_period)

        # Create a CrossOver indicator: > 0 indicates fast > slow, < 0 indicates fast < slow
        self.crossover = bt.indicators.CrossOver(self.fast_moving_average, self.slow_moving_average)

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buy_price = None
        self.comm = None

    def log(self, txt, dt=None):
        """Logger function for this strategy."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """
        Notification of an order status change.
        This method is called by Cerebro whenever the status of an order changes.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return

        # Check if an order has been completed (or rejected/canceled)
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )
                self.buy_price = order.executed.price
                self.comm = order.executed.comm
            elif order.issell():
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )
            self.bar_executed = len(self) # Store the bar index when the order was executed
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset the order reference as it's no longer active
        self.order = None

    def notify_trade(self, trade):
        """
        Notification of a trade outcome.
        This method is called by Cerebro when a trade is closed.
        """
        if not trade.isclosed:
            return # Only interested in closed trades

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        """
        This method is called by Cerebro for each bar of data.
        It contains the main trading logic.
        """
        # If an order is pending, do nothing until it's completed
        if self.order:
            return

        # Check if we are currently in the market (have an open position)
        if not self.position:  # Not in the market
            # Buy condition: Fast SMA crosses above Slow SMA
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                # Place a buy order
                self.order = self.buy()
        else:  # Already in the market (have an open position)
            # Sell condition: Fast SMA crosses below Slow SMA
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                # Place a sell order
                self.order = self.sell()


# --- Flask Routes ---
@app.route('/')
def index():
    """
    Renders the main page of the application with the form for uploading data
    and setting strategy parameters.
    """
    return render_template('index.html', results=None)

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """
    Handles the POST request to run the backtest.
    It processes the uploaded CSV file and user-defined parameters,
    then executes the Backtrader strategy and returns the results.
    """
    # Check if a file was uploaded in the request
    if 'data_file' not in request.files:
        return render_template('index.html', results={"message": "No file part in the request.", "error": True})

    file = request.files['data_file']
    # Check if the file input field was empty
    if file.filename == '':
        return render_template('index.html', results={"message": "No selected file.", "error": True})

    # Ensure a file was actually provided
    if not file:
        return render_template('index.html', results={"message": "File upload failed.", "error": True})

    try:
        # Read the uploaded CSV data into a Pandas DataFrame
        # decode('utf-8') is used because file.read() returns bytes
        data_io = io.StringIO(file.read().decode('utf-8'))
        # Read CSV, parse 'Date' column as datetime and set it as index
        df = pd.read_csv(data_io, parse_dates=True, index_col='Date')
        # Ensure the index is a datetime object, converting if necessary
        df.index = pd.to_datetime(df.index)

        # Get strategy parameters from the form.
        # Use .get() with default values to prevent errors if parameters are missing.
        fast_period = int(request.form.get('fast_period', 10))
        slow_period = int(request.form.get('slow_period', 30))

        # Initialize Cerebro engine, which is the core of Backtrader
        cerebro = bt.Cerebro()

        # Add the SMA Crossover strategy to Cerebro with user-defined parameters
        cerebro.addstrategy(SMACrossover, fast_period=fast_period, slow_period=slow_period)

        # Set the starting cash for the backtest
        cerebro.broker.setcash(100000.0) # Start with $100,000

        # Add the data feed to Cerebro.
        # bt.feeds.PandasData is used to feed a Pandas DataFrame to Backtrader.
        # Pass the DataFrame using the 'datanames' keyword argument,
        # and map columns using their string names from the CSV.
        data = bt.feeds.PandasData(
            dataname=df,  # Pass the DataFrame using 'dataname'
            datetime=None, # Date is already the index
            open='Open',        # Map to 'Open' column in DataFrame
            high='High',        # Map to 'High' column in DataFrame
            low='Low',          # Map to 'Low' column in DataFrame
            close='Close',      # Map to 'Close' column in DataFrame
            volume='Volume',    # Map to 'Volume' column in DataFrame
            openinterest='OpenInterest', # Map to 'OpenInterest' column in DataFrame
        )
        cerebro.adddata(data)

        # Set commission for trades (e.g., 0.1% per trade)
        cerebro.broker.setcommission(commission=0.001)

        # Record the initial portfolio value before running the backtest
        initial_portfolio_value = cerebro.broker.getvalue()

        # Run the backtest. This executes the strategy over the data.
        print("Running backtest...")
        strategies = cerebro.run() # The run method returns a list of strategy instances
        print("Backtest finished.")

        # Get the final portfolio value after the backtest
        final_portfolio_value = cerebro.broker.getvalue()
        profit = final_portfolio_value - initial_portfolio_value
        # Calculate profit percentage, handle division by zero if initial_portfolio_value is 0
        profit_percent = (profit / initial_portfolio_value) * 100 if initial_portfolio_value else 0

        # Prepare results to be displayed in the HTML template
        results = {
            'initial_value': f"${initial_portfolio_value:,.2f}", # Format as currency
            'final_value': f"${final_portfolio_value:,.2f}",
            'profit': f"${profit:,.2f}",
            'profit_percent': f"{profit_percent:,.2f}%",
            'fast_period': fast_period,
            'slow_period': slow_period,
            'message': "Backtest completed successfully!",
            'error': False # Indicate no error
        }

    except Exception as e:
        # Catch any exceptions during the process and prepare an error message
        results = {
            'message': f"An error occurred during backtesting: {e}",
            'error': True # Indicate an error occurred
        }

    # Render the index page again, but this time with the backtest results
    return render_template('index.html', results=results)

# Run the Flask application in debug mode (for development)
if __name__ == '__main__':
    app.run(debug=True)
