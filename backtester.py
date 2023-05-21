
### https://medium.com/coinmonks/here-is-how-i-coded-a-multi-asset-backtester-in-python-ff39db1751f7
import pandas as pd
import os
import matplotlib.pyplot as plt
from ta.momentum import rsi
from ta.trend import ema_indicator
import numpy as np
from pathlib import Path


class Position():
    def __init__(self, pair, side, size, fee_rate, ohlc):
        self.pair = pair
        self.side = side
        self.fee_rate = fee_rate
        self.entry_time = ohlc['close_time'].iloc[-1]
        self.entry_price = ohlc['Close'].iloc[-1]
        self.size = (1 - fee_rate) * size
        self.fees = fee_rate * size
        self.pnl = self.fees
        self.unrealized_pnl = self.fees
        self.status = 'open'
        self.highest_price_seen = self.entry_price
        self.lowest_price_seen = self.entry_price
        self.drawdown = 0
        self.max_drawdown = 0
        return

    def update_stats(self, ohlc):
        if self.status != 'open':
            raise Exception("Cannot update stats of a closed position")
        price = ohlc['Close'].iloc[-1]
        if price > self.highest_price_seen:
            self.highest_price_seen = price
        if price < self.lowest_price_seen:
            self.lowest_price_seen = price
        if self.side == 'long':
            self.unrealized_pnl = self.fees + (1 - self.fee_rate) * self.size * (price - self.entry_price) / self.entry_price
            self.drawdown = self.highest_price_seen - price
        if self.side == 'short':
            self.unrealized_pnl = self.fees + (1 - self.fee_rate) * self.size * (self.entry_price - price) / self.entry_price
            self.drawdown = price - self.lowest_price_seen
        self.max_drawdown = np.max((self.max_drawdown, self.drawdown))
        return

    def close(self, ohlc):
        self.update_stats(ohlc)
        self.exit_time = ohlc['close_time'].iloc[-1]
        self.exit_price = ohlc['Close'].iloc[-1]
        if self.side == 'long':
            self.pnl += (1 - self.fee_rate) * self.size * (self.exit_price - self.entry_price) / self.entry_price
            self.fees += self.fee_rate * self.size * (self.exit_price - self.entry_price) / self.entry_price
        if self.side == 'short':
            self.pnl += (1 - self.fee_rate) * self.size * (self.entry_price - self.exit_price) / self.entry_price
            self.fees += self.fee_rate * self.size * (self.entry_price - self.exit_price) / self.entry_price
        self.pnl_pct = self.pnl / (self.size / (1 - self.fee_rate))
        self.success = 1 if self.pnl > 0 else 0
        self.status = 'closed'
        new_cash = self.size + self.pnl
        return new_cash


class Strategy():
    def __init__(self, risk_factor):
        self.risk_factor = risk_factor
        self.required_data_length = 100
        return

    def __str__(self):
        description = "Demo EMA Strategy"
        description += "\n Buy long when the fast EMA is above the slow one;"
        description += "\n Sell short when the fats EMA is under the slow one;"
        return description

    def position_size(self, ohlc, portfolio_value):
        pct_changes = (ohlc['Close'] - ohlc['Close'].shift(1)) / ohlc['Close'].shift(1)
        std_dev = np.std(pct_changes.values[-20:])
        size = self.risk_factor * portfolio_value / (10*std_dev)
        return size

    def compute_indicators(self, ohlc):
        if ohlc.shape[0] < self.required_data_length:
            raise Exception("Length of passed OHLC data too small")
        ohlc['rsi'] = rsi(ohlc['Close'], 3)
        ohlc['ema_fast'] = ema_indicator(ohlc['Close'], 10)
        ohlc['ema_slow'] = ema_indicator(ohlc['Close'], 20)
        ohlc['uptrend'] = np.where(ohlc['ema_fast'] >= ohlc['ema_slow'], True, False)
        ohlc['downtrend'] = np.where(ohlc['ema_fast'] < ohlc['ema_slow'], True, False)
        return ohlc

    def open_long(self, ohlc):
        long_signal = (ohlc['uptrend'].iloc[-1] and not ohlc['uptrend'].iloc[-2])
        return long_signal

    def open_short(self, ohlc):
        short_signal = (ohlc['downtrend'].iloc[-1] and not ohlc['downtrend'].iloc[-2])
        return short_signal

    def close_position(self, ohlc, position):
        if position.status != 'open':
            raise Exception('Unable to close a closed position.')
        if (position.side == 'long') and (not ohlc['uptrend'].iloc[-1]):
            return True
        if (position.side == 'short') and (not ohlc['downtrend'].iloc[-1]):
            return True
        return False

class PerformanceTracker():
    def __init__(self):
        root = Path(os.getcwd())
        result_dir_path = root / 'backtest_results'
        if os.path.exists(result_dir_path):
            for file_name in os.listdir(result_dir_path):
                file_path = result_dir_path / file_name
                os.remove(file_path)
            os.rmdir(result_dir_path)
        os.mkdir(result_dir_path)
        self.paths = {
            'portfolio': result_dir_path / 'portfolio_evolution.csv',
            'closed_positions': result_dir_path / 'closed_positions.csv'
        }
        self.PORTFOLIO = list()
        self.CLOSED_POSITIONS = list()
        self.dataframes_created = False
        return

    def track_portfolio_values(self, backtest):
        """Track portfolio's values"""
        portfolio_value = backtest.available_capital
        realized_portfolio_value = backtest.available_capital
        available_capital = backtest.available_capital
        for position in backtest.open_positions:
            realized_portfolio_value += position.size
            portfolio_value += position.size
            portfolio_value += position.unrealized_pnl
        self.PORTFOLIO.append({
            'close_time': backtest.current_ts,
            'available_capital': available_capital,
            'portfolio_value': portfolio_value,
            'realized_portfolio_value': realized_portfolio_value
        })
        return

    def save_closed_position(self, position):
        """Save a position that has been closed."""
        self.CLOSED_POSITIONS.append({
            'pair': position.pair,
            'side': position.side,
            'fee_rate': position.fee_rate,
            'entry_time': position.entry_time,
            'exit_time': position.exit_time,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'size': position.size,
            'pnl': position.pnl,
            'pnl%': position.pnl_pct,
            'fees': position.fees,
            'success': position.success,
            'status': position.status,
            'highest_price_seen': position.highest_price_seen,
            'lowest_price_seen': position.lowest_price_seen,
            'drawdown': position.drawdown,
            'max_drawdown': position.max_drawdown
        })
        return

    def to_dataframes(self):
        """Build a nice dataframe out of all the tracked statistics"""
        self.DF_PORTFOLIO = pd.DataFrame(self.PORTFOLIO)
        self.DF_CLOSED_POSITIONS = pd.DataFrame(self.CLOSED_POSITIONS)
        self.DF_PORTFOLIO.to_csv(self.paths['portfolio'], sep=',', index=False)
        self.DF_CLOSED_POSITIONS.to_csv(self.paths['closed_positions'], sep=',', index=False)
        self.dataframes_created = True
        return

    def plot_portfolio_evolution(self):
        """Plot portfolio realized and unrealized values, with available capital over time."""
        if not self.dataframes_created:
            raise Exception("A backtest must be lead before plotting any portfolio evolution.")
        start_date = self.DF_PORTFOLIO['close_time'].min()
        end_date = self.DF_PORTFOLIO['close_time'].max()
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)
        # ax.plot(self.DF_PORTFOLIO['close_time'], self.DF_PORTFOLIO['available_capital'], linewidth=0.8, color='deepskyblue', label='available capital')
        ax.plot(self.DF_PORTFOLIO['close_time'], self.DF_PORTFOLIO['realized_portfolio_value'], linewidth=0.8, color='darkblue', label='realized portfolio value')
        ax.plot(self.DF_PORTFOLIO['close_time'], self.DF_PORTFOLIO['portfolio_value'], linewidth=0.8, color='red', label='unrealized portfolio value')
        ax.set_xlabel('close time', fontsize=15)
        ax.set_ylabel('quote currency', fontsize=15)
        ax.set_title(f'Evolution of portfolio during backtest from {start_date} to {end_date}', fontsize=18)
        ax.legend()
        plt.show()
        return

    def plot_cumulated_profits(self):
        """Plot the cumulated profits of each pair, and of the whole universe."""
        if not self.dataframes_created:
            raise Exception("A backtest must be lead before plotting any profits evolution.")
        start_date = self.DF_PORTFOLIO['close_time'].min()
        end_date = self.DF_PORTFOLIO['close_time'].max()
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)
        # pair,side,fee_rate,entry_time,exit_time,entry_price,exit_price,size,pnl,pnl%,fees,success,status,highest_price_seen,lowest_price_seen,drawdown,max_drawdown
        for pair in list(set(self.DF_CLOSED_POSITIONS['pair'].values.tolist())):
            pair_df = self.DF_CLOSED_POSITIONS[self.DF_CLOSED_POSITIONS['pair'] == pair].copy()
            ax.plot(pair_df['exit_time'], pair_df['pnl'].cumsum(), linewidth=0.7, label=pair)
        ax.plot(self.DF_CLOSED_POSITIONS['exit_time'], self.DF_CLOSED_POSITIONS['pnl'].cumsum(), linewidth=1.2, color='darkblue', label='cumulated profits (all pairs)')
        ax.plot(self.DF_CLOSED_POSITIONS['exit_time'], [0]*len(self.DF_CLOSED_POSITIONS['exit_time']), linewidth=0.8, color='black')
        ax.set_xlabel('close time', fontsize=15)
        ax.set_ylabel('quote currency', fontsize=15)
        ax.set_title(f'Cumulated profits during backtest from {start_date} to {end_date}', fontsize=18)
        ax.legend()
        plt.show()
        return
    
    def compute_backtest_statistics(self, timedelta):
        """Compute win rate, avg_gain, avg_loss, expected_return, sharpe ratio, sortino ratio, max drawdown."""
        if not self.dataframes_created:
            raise Exception("A backtest must be lead before computing backtest statistics.")
        TRADES = self.DF_CLOSED_POSITIONS.copy()  
        LONG = self.DF_CLOSED_POSITIONS[self.DF_CLOSED_POSITIONS['side'] == 'long'].copy()
        SHORT = self.DF_CLOSED_POSITIONS[self.DF_CLOSED_POSITIONS['side'] == 'short'].copy()
        one_year_duration = pd.Timedelta(days=365)
        # Annualized Sharpe Ratio
        pf_mean_returns = self.DF_PORTFOLIO['portfolio_value'].pct_change().mean()
        pf_std_returns = self.DF_PORTFOLIO['portfolio_value'].pct_change().std()
        sharpe_ratio = pf_mean_returns / pf_std_returns
        sharpe_ratio *= np.sqrt(one_year_duration / timedelta)
        sharpe_ratio = round(sharpe_ratio, 2)
        # Annualized Sortino Ratio
        positive_returns = self.DF_PORTFOLIO['portfolio_value'].pct_change()
        positive_returns = positive_returns[positive_returns > 0].copy()
        pf_std_positive_returns = positive_returns.std()
        sortino_ratio = pf_mean_returns / pf_std_positive_returns
        sortino_ratio *= np.sqrt(one_year_duration / timedelta)
        sortino_ratio = round(sortino_ratio, 2)
        # Win Rates
        global_win_rate = round(100 * TRADES['success'].sum() / TRADES.shape[0], 2) if TRADES.shape[0] != 0 else 0
        long_win_rate = round(100 * LONG['success'].sum() / LONG.shape[0], 2) if LONG.shape[0] != 0 else 0
        short_win_rate = round(100 * SHORT['success'].sum() / SHORT.shape[0], 2) if SHORT.shape[0] != 0 else 0
        # Average Gains
        global_avg_gain = round(100 * TRADES[TRADES['success'] == 1]['pnl%'].mean(), 2)
        long_avg_gain = round(100 * LONG[LONG['success'] == 1]['pnl%'].mean(), 2)
        short_avg_gain = round(100 * SHORT[SHORT['success'] == 1]['pnl%'].mean(), 2)
        # Average Gains
        global_avg_loss = round(100 * TRADES[TRADES['success'] == 0]['pnl%'].mean(), 2)
        long_avg_loss = round(100 * LONG[LONG['success'] == 0]['pnl%'].mean(), 2)
        short_avg_loss = round(100 * SHORT[SHORT['success'] == 0]['pnl%'].mean(), 2)
        # Expected Returns
        global_exp_return = round(100 * ((global_win_rate/100) * (global_avg_gain/100) + (1 - global_win_rate/100) * (global_avg_loss/100)), 2)
        long_exp_return = round(100 * ((long_win_rate/100) * (long_avg_gain/100) + (1 - long_win_rate/100) * (long_avg_loss/100)), 2)
        short_exp_return = round(100 * ((short_win_rate/100) * (short_avg_gain/100) + (1 - short_win_rate/100) * (short_avg_loss/100)), 2)
        # Max drawdowns
        max_achieved_value = np.array([np.max(self.DF_PORTFOLIO['portfolio_value'].iloc[:i]) for i in range(1, self.DF_PORTFOLIO.shape[0])])
        max_drawdowns = round(np.max(-100 * (self.DF_PORTFOLIO['portfolio_value'].iloc[1:].values / max_achieved_value - 1)), 2)
        
        print("\n----- BACKTEST RESULTS -----")
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Sortino Ratio: {sortino_ratio}")
        print(f"Max Drawdown: {max_drawdowns}%")
        print()
        print(f"Global win rate: {global_win_rate}%")
        print(f"Global average gain: {global_avg_gain}%")
        print(f"Global average loss: {global_avg_loss}%")
        print(f"Global Expected Return: {global_exp_return}%")
        print()
        print(f"Long win rate: {long_win_rate}%")
        print(f"Long average gain: {long_avg_gain}%")
        print(f"Long average loss: {long_avg_loss}%")
        print(f"Long Expected Return: {long_exp_return}%")
        print()
        print(f"Short win rate: {short_win_rate}%")
        print(f"Short average gain: {short_avg_gain}%")
        print(f"Short average loss: {short_avg_loss}%")
        print(f"Short Expected Return: {short_exp_return}%")
        
        print("----------------------------")

        return

class Backtester():
    def __init__(self, strategy, start_ts, end_ts, settings):
        """
        settings = {
            'timeframe',
            'universe',
            'initial_portfolio_value',
            'fee_rate'
        }
        """
        self.universe = settings['universe']
        self.timeframe = settings['timeframe']
        self.timedelta = pd.Timedelta(self.timeframe)
        self.portfolio_value = settings['initial_portfolio_value']
        self.available_capital = settings['initial_portfolio_value']
        self.fee_rate = settings['fee_rate']
        self.strategy = strategy
        self.tracker = PerformanceTracker()
        self.open_positions = []
        self.closed_positions = []
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.current_ts = start_ts
        return

    def read_csv(self, pair):
        root = Path(os.getcwd())
        data_path = root.parent / 'learn/data'
#        path = data_path / f'{pair.lower()}{self.timeframe}.csv'
        path = data_path / f'{pair.lower()}.csv'
        df = pd.read_csv(path, sep=',')
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['close_time'] = pd.to_datetime(df['close_time'])
        df = df[df['close_time'] <= self.current_ts]
        df.index = [i for i in range(df.shape[0])]
        df = df[[
            'open_time', 'close_time', 
            'Open', 'High', 'Low', 'Close',
            'Volume'
        ]].iloc[-3 * self.strategy.required_data_length:].copy()
        df.index = [i for i in range(df.shape[0])]
        return df

    def load_continuation_data(self):
        """Gather data up to the current time."""
        data = dict()
        for pair in self.universe:
            data[pair] = self.read_csv(pair)
            # print(f"data shape: {data[pair].shape}")
            data[pair] = self.strategy.compute_indicators(data[pair])
        return data

    def manage_open_position(self, position, pair_data):
        """Where to close any existing positions."""
        if self.strategy.close_position(pair_data, position):
            new_cash = position.close(pair_data)
            self.closed_positions.append(position)
            self.tracker.save_closed_position(position)
            self.available_capital += new_cash
        return position

    def track_portfolio_value(self):
        """Sum up available capital, plus all positions'unrealized pnl and size"""
        self.portfolio_value = self.available_capital
        for position in self.open_positions:
            self.portfolio_value += position.size
            self.portfolio_value += position.unrealized_pnl
        return

    def cyclic_process(self):
        """Process to be done during each iteration of the backtest"""
        data = self.load_continuation_data()
        # Manage open positions
        still_open_positions = []
        for position in self.open_positions:
            position.update_stats(data[position.pair])
            position = self.manage_open_position(position, data[position.pair])
            if position.status == 'open':
                still_open_positions.append(position)
        self.open_positions = still_open_positions
        # Open new positions
        for pair in self.universe:
            if self.strategy.open_long(data[pair]):
                size = self.strategy.position_size(data[pair], self.portfolio_value)
                if size <= self.available_capital: 
                    print(f"Buy: {size}\n{data[pair].tail(1)}")                   
                    self.open_positions.append(
                        Position(
                            pair=pair,
                            side='long',
                            size=size,
                            fee_rate=self.fee_rate,
                            ohlc=data[pair]
                        )
                    )
                    self.available_capital -= size
            
            if self.strategy.open_short(data[pair]):
                size = self.strategy.position_size(data[pair], self.portfolio_value)
                if size <= self.available_capital:                    
                    self.open_positions.append(
                        Position(
                            pair=pair,
                            side='short',
                            size=size,
                            fee_rate=self.fee_rate,
                            ohlc=data[pair]
                        )
                    )
                    print(f"Sell: {size}\n{data[pair].tail(1)}") 
                    self.available_capital -= size
        # Increment current time and update portfolio value
        self.tracker.track_portfolio_values(self)
        self.track_portfolio_value()
        self.current_ts += self.timedelta
        return
        
    def execute(self):
        """Execute backtest."""
        while self.current_ts <= self.end_ts:
            print(f" --> self.current_ts: {self.current_ts}")
            self.cyclic_process()
        self.tracker.to_dataframes()
        self.tracker.plot_portfolio_evolution()
        self.tracker.plot_cumulated_profits()
        self.tracker.compute_backtest_statistics(self.timedelta)
        return
    
settings = {
    'timeframe': '1d',
    # 'universe': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
    'universe': ['tsla'],
    'initial_portfolio_value': 1000,
    'fee_rate': 0.0004
}
start_ts = pd.Timestamp("2017-10-01 00:00:00")
end_ts = pd.Timestamp("2023-04-30 00:00:00")
strategy = Strategy(risk_factor=0.01)
backtester = Backtester(strategy, start_ts, end_ts, settings)
backtester.execute()
