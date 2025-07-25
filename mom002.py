from algo_deploy.strategy_module import LiveStrategy, Indicator, extract_ochlv
from typing import Optional, Dict
import signal
import time

class MOM002(LiveStrategy):
    """
    MOM002: Momentum Strategy - v002 - add ob imbalance entry
    """

    def __init__(self, strategy_name: str, params: Dict, account_id: str, exchange: str, base: str, quote: str,
                 category: str = 'futures', demo: bool = True):
        super().__init__(strategy_name, account_id, exchange, base, quote, category, demo)
        self.add_params(params)
        self.stop_flag = False

        # Setup signal handler once on init
        signal.signal(signal.SIGINT, self._signal_handler)

        self.latest_entry_ts: int = 0
        self.init_balance: float = 0
        self.open_order_ids: set = set()
        
        self.is_buy = False
        self.is_sell = False

    def add_params(self, params: dict):
        """Update strategy parameters."""
        
        # --- Core strategy logic ---
        self.period = params.get('period', 30)
        self.alpha = params.get('alpha', 0.0001)
        self.ib_theshold = params.get('ib_theshold', 1)
        self.z_threshold = params.get('z_threshold', 0.3)
        self.value_entry = params.get('value_entry', 10)
        self.stop_loss_threshold = params.get('stop_loss_threshold', 0.2)

        # --- Execution / Order parameters ---
        self.max_orders = params.get('max_orders', 5)
        self.refresh_order = params.get('refresh_order', 60)
        self.refresh_position = params.get('refresh_position', 60 * 60)

        # --- Time settings ---
        self.interval = params.get('interval', '1m')
        self.sleep_time = params.get('sleep_time', 60)

    def get_entry(self, interval: str = '15m') -> float:
        """Generate entry signal based on Z-score, Kalman mean
        return: Signal
        
        """
        signal = 0
        try:
            candles = self.get_candles(self.base, self.quote, interval=interval, use_redis=False)
            o, c, h, l, v = extract_ochlv(candles)
            n = len(c)

            if n < self.period + 2:
                self.logger.warning("Not enough data for indicator calculations.")
                return signal

            # Indicators
            kalman = Indicator.kalman_filter_mean(c, adjust_factor=2)

            # Use EWM std for smoother Z-score
            std = Indicator.rolling_std(c, self.period)
            z_score = Indicator.z_score(c, kalman, std)

            _, trend = Indicator.trend_detect(h, l, c, self.period)
            curr_price = self.get_imbalance_price(self.base, self.quote, 0.01)

            price_signal = 1 if curr_price >= c[-1] else -1
            trend_signal = trend[-1] == -1

            ob_i = self.get_imbalance_ob(self.base, self.quote, 0.01)
            threshold = self.z_threshold

            if (z_score[-1] > threshold and price_signal == 1 and trend_signal) or ob_i >= self.ib_theshold:
                signal = 1
                
            elif z_score[-1] < -threshold and price_signal == -1 and trend_signal or ob_i <= self.ib_theshold * -1:
                signal = -1

            self.logger.info(
                f"Signal: {signal}, Z: {z_score[-1]:.3f}, Trend: {trend[-1]}, Ob imbalance: {round(ob_i, 2)}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error in get_entry_price: {e} | Line: {e.__traceback__.tb_lineno}")
            return signal

    def calculate_position_size(self, value_entry: float = 10, price: Optional[float] = None) -> float:
        """Calculate position size based on available quote balance and risk pct."""
        balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0) * 10

        if self.value_entry > balance:
            self.logger.warning("Insufficient balance to open a new position.")
            return 0.0

        price = price or self.get_price(self.base, self.quote)
        if not price or price <= 0:
            self.logger.warning("Invalid price for sizing.")
            return 0.0
        
        quantity = round(value_entry / price, 6)

        return quantity
    
    def handle_orders(self):
        """Cancel stale orders based on time thresholds."""
        now = time.time() * 1000  # current timestamp in ms
        self._update_order()

        if not self.position.orders:
            return

        order_ids = self.position.orders

        for order_id, fill_info in list(order_ids.items()):
            status = fill_info.get("status")
            entry_time = int(fill_info.get("orderUpdateTime", 0))

            if not order_id or not entry_time:
                continue  # skip if missing key data

            elapsed = now - entry_time

            if status in ['NEW', 'UNKNOWN']:
                self.open_order_ids.add(order_id)

                if elapsed > self.refresh_order * 1000:
                    self.position.close(self, order_id)
                    self.open_order_ids.discard(order_id)
                    self.position.orders.pop(order_id, None)
                    self.logger.info(f"Cancelled stale unfilled order: {order_id}")
                
            elif status in ['PARTIALLY_FILLED', 'FILLED']:
                self.latest_entry_ts = max(self.latest_entry_ts, entry_time)
                self.open_order_ids.discard(order_id)

                if elapsed > self.refresh_position * 1000:
                    self.position.close(self, order_id)
                    self.open_order_ids.discard(order_id)
                    self.position.orders.pop(order_id, None) 
                    self.logger.info(f"Cancelled stale filled order: {order_id}")

    def execute_trade(self, signal: int, price: float, quantity: float,
                      order_type: str = 'limit', force: str = 'GTC') -> bool:
        """
        Ensures delay between repeated executions using latest_entry_ts.
        """
        now_ms = time.time() * 1000
        wait_ms = self.refresh_order // self.max_orders * 1000

        if (now_ms - self.latest_entry_ts) < wait_ms:
            wait_left = wait_ms - (now_ms - self.latest_entry_ts)
            self.logger.info(f"[execute_trade] Skipping trade. Wait {wait_left / 1000:.2f}s more.")
            return False

        self.latest_entry_ts = now_ms
        if signal == 1:
            price = self.get_price(self.base, self.quote, 'bid') * (1 - self.alpha)
        else: 
            price = self.get_price(self.base, self.quote, 'ask') * (1 + self.alpha)

        if signal == 1 and not self.is_buy:
           result = self.buy(quantity, price, order_type=order_type, force=force)
           self.is_sell = False
           self.is_buy = True
           return result
       
        elif signal == -1 and signal == -1:
            result = self.sell(quantity, price, order_type=order_type, force=force)
            self.is_sell = True
            self.is_buy = False
            return result

    def close_all_remaining_positions(self):
        """Close all remaining positions forcefully."""
        self.logger.info("Closing all remaining orders and positions forcefully...")
        self.position.cancel_all_orders(self, force_close=True)
        self.position.close_all_positions(self, force_close=True)

    def handle_stop_loss(self):
        """Check stop loss condition."""
        curr_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)
        if curr_balance < self.init_balance * (1 - self.stop_loss_threshold):
            self.logger.warning("Stop Loss triggered!")
            self.stop_flag = True
            self.handle_exit()

    def run(self):
        """Run main strategy loop."""
        self.close_all_remaining_positions()
        self.logger.info("Starting Strategy...")
        self.init_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)
        try:
            while not self.stop_flag:
                signal = self.get_entry(self.interval)

                self.handle_orders()
                self.handle_stop_loss()

                if signal != 0 and len(self.open_order_ids) < self.max_orders:
                    price = self.get_price(self.base, self.quote)
                    size = self.calculate_position_size(self.value_entry, price)
                    if size > 0:
                        self.execute_trade(signal, price, size, order_type='limit')
                    else:
                        self.logger.warning("Zero position size, skipping trade.")
                else:
                    # self.logger.info("No trade or max position reached.")
                    pass

                time.sleep(self.sleep_time)

        except Exception as e:
            self.logger.exception(f"Run error: {e}")

        finally:
            self.handle_exit()
            self.logger.info("Strategy stopped cleanly.")
