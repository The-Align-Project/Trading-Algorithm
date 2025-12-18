"""Options trading module with various strategies and Greeks calculation."""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from data_structures import OptionsSignal, OptionLeg, OptionType, OptionAction, OptionsPosition
from config import YF_AVAILABLE, SCIPY_AVAILABLE, TradingConfig, OPTIONS_STRATEGIES

if SCIPY_AVAILABLE:
    from scipy.stats import norm

class BlackScholes:
    """Black-Scholes model for options pricing"""

    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price"""
        if not SCIPY_AVAILABLE:
            return 0.0

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(0, price)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price"""
        if not SCIPY_AVAILABLE:
            return 0.0

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(0, price)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option delta"""
        if not SCIPY_AVAILABLE:
            return 0.5

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma"""
        if not SCIPY_AVAILABLE:
            return 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option theta (per day)"""
        if not SCIPY_AVAILABLE:
            return 0.0

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        if option_type.lower() == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(d2))
        else:  # put
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * norm.cdf(-d2))

        return theta / 365  # Convert to daily theta

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega"""
        if not SCIPY_AVAILABLE:
            return 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega / 100  # Vega per 1% change in volatility

class OptionsAnalyzer:
    """Analyzes options opportunities and manages options strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = 0.05  # 5% annual risk-free rate

    def calculate_implied_volatility(self, data: pd.DataFrame, window: int = 30) -> float:
        """Calculate implied volatility from price history"""
        if len(data) < window:
            return 0.3  # Default 30% volatility

        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return volatility

    def find_optimal_strikes(self, current_price: float, volatility: float,
                            strategy: str) -> Dict[str, float]:
        """Find optimal strike prices for a strategy"""
        strikes = {}

        if strategy == 'long_call':
            # Slightly OTM call
            strikes['call'] = current_price * 1.05

        elif strategy == 'long_put':
            # Slightly OTM put
            strikes['put'] = current_price * 0.95

        elif strategy == 'bull_call_spread':
            # Buy ATM, sell OTM
            strikes['buy_call'] = current_price
            strikes['sell_call'] = current_price * 1.10

        elif strategy == 'bear_put_spread':
            # Buy ATM, sell OTM
            strikes['buy_put'] = current_price
            strikes['sell_put'] = current_price * 0.90

        elif strategy == 'iron_condor':
            # Sell OTM put/call, buy further OTM put/call
            strikes['buy_put'] = current_price * 0.90
            strikes['sell_put'] = current_price * 0.95
            strikes['sell_call'] = current_price * 1.05
            strikes['buy_call'] = current_price * 1.10

        return strikes

    def calculate_strategy_payoff(self, strategy_legs: List[OptionLeg],
                                  spot_prices: np.ndarray) -> np.ndarray:
        """Calculate payoff diagram for options strategy"""
        total_payoff = np.zeros_like(spot_prices)

        for leg in strategy_legs:
            premium = leg.premium
            strike = leg.strike
            quantity = leg.quantity

            if leg.option_type == OptionType.CALL:
                # Call payoff
                intrinsic_value = np.maximum(spot_prices - strike, 0)
            else:
                # Put payoff
                intrinsic_value = np.maximum(strike - spot_prices, 0)

            # Buy adds cost, sell adds credit
            if leg.action == OptionAction.BUY:
                leg_payoff = (intrinsic_value - premium) * quantity
            else:  # SELL
                leg_payoff = (premium - intrinsic_value) * quantity

            total_payoff += leg_payoff

        return total_payoff

    def create_long_call(self, symbol: str, current_price: float,
                        volatility: float, dte: int = 30) -> Optional[OptionsSignal]:
        """Create long call options signal"""
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available. Cannot price options.")
            return None

        strike = current_price * 1.05  # 5% OTM
        expiration = datetime.now() + timedelta(days=dte)
        T = dte / 365

        # Calculate option price
        premium = BlackScholes.call_price(
            current_price, strike, T, self.risk_free_rate, volatility
        )

        leg = OptionLeg(
            option_type=OptionType.CALL,
            action=OptionAction.BUY,
            strike=strike,
            expiration=expiration,
            quantity=1,
            premium=premium
        )

        # Calculate max risk/profit
        max_risk = premium * 100  # Per contract
        max_profit = float('inf')  # Unlimited for long call
        breakeven = strike + premium

        return OptionsSignal(
            symbol=symbol,
            strategy_name='long_call',
            legs=[leg],
            confidence=0.7,
            max_risk=max_risk,
            max_profit=max_profit,
            breakeven_points=[breakeven]
        )

    def create_bull_call_spread(self, symbol: str, current_price: float,
                                volatility: float, dte: int = 30) -> Optional[OptionsSignal]:
        """Create bull call spread"""
        if not SCIPY_AVAILABLE:
            return None

        buy_strike = current_price
        sell_strike = current_price * 1.10
        expiration = datetime.now() + timedelta(days=dte)
        T = dte / 365

        # Calculate premiums
        buy_premium = BlackScholes.call_price(
            current_price, buy_strike, T, self.risk_free_rate, volatility
        )
        sell_premium = BlackScholes.call_price(
            current_price, sell_strike, T, self.risk_free_rate, volatility
        )

        legs = [
            OptionLeg(OptionType.CALL, OptionAction.BUY, buy_strike, expiration, 1, buy_premium),
            OptionLeg(OptionType.CALL, OptionAction.SELL, sell_strike, expiration, 1, sell_premium)
        ]

        net_debit = (buy_premium - sell_premium) * 100
        max_risk = net_debit
        max_profit = (sell_strike - buy_strike - (buy_premium - sell_premium)) * 100
        breakeven = buy_strike + (buy_premium - sell_premium)

        return OptionsSignal(
            symbol=symbol,
            strategy_name='bull_call_spread',
            legs=legs,
            confidence=0.75,
            max_risk=max_risk,
            max_profit=max_profit,
            breakeven_points=[breakeven]
        )

    def create_iron_condor(self, symbol: str, current_price: float,
                          volatility: float, dte: int = 45) -> Optional[OptionsSignal]:
        """Create iron condor for neutral markets"""
        if not SCIPY_AVAILABLE:
            return None

        # Define strikes
        buy_put_strike = current_price * 0.90
        sell_put_strike = current_price * 0.95
        sell_call_strike = current_price * 1.05
        buy_call_strike = current_price * 1.10

        expiration = datetime.now() + timedelta(days=dte)
        T = dte / 365

        # Calculate premiums
        buy_put_premium = BlackScholes.put_price(
            current_price, buy_put_strike, T, self.risk_free_rate, volatility
        )
        sell_put_premium = BlackScholes.put_price(
            current_price, sell_put_strike, T, self.risk_free_rate, volatility
        )
        sell_call_premium = BlackScholes.call_price(
            current_price, sell_call_strike, T, self.risk_free_rate, volatility
        )
        buy_call_premium = BlackScholes.call_price(
            current_price, buy_call_strike, T, self.risk_free_rate, volatility
        )

        legs = [
            OptionLeg(OptionType.PUT, OptionAction.BUY, buy_put_strike, expiration, 1, buy_put_premium),
            OptionLeg(OptionType.PUT, OptionAction.SELL, sell_put_strike, expiration, 1, sell_put_premium),
            OptionLeg(OptionType.CALL, OptionAction.SELL, sell_call_strike, expiration, 1, sell_call_premium),
            OptionLeg(OptionType.CALL, OptionAction.BUY, buy_call_strike, expiration, 1, buy_call_premium)
        ]

        # Calculate risk/reward
        net_credit = (sell_put_premium + sell_call_premium - buy_put_premium - buy_call_premium) * 100
        put_wing_width = (sell_put_strike - buy_put_strike) * 100
        call_wing_width = (buy_call_strike - sell_call_strike) * 100
        max_risk = max(put_wing_width, call_wing_width) - net_credit
        max_profit = net_credit

        breakeven_lower = sell_put_strike - (net_credit / 100)
        breakeven_upper = sell_call_strike + (net_credit / 100)

        return OptionsSignal(
            symbol=symbol,
            strategy_name='iron_condor',
            legs=legs,
            confidence=0.65,
            max_risk=max_risk,
            max_profit=max_profit,
            breakeven_points=[breakeven_lower, breakeven_upper]
        )

    def generate_options_signal(self, symbol: str, data: pd.DataFrame,
                               market_outlook: str) -> Optional[OptionsSignal]:
        """Generate options signal based on market outlook"""
        if len(data) < 30:
            return None

        current_price = data['Close'].iloc[-1]
        volatility = self.calculate_implied_volatility(data)

        # Choose strategy based on outlook
        if market_outlook == "BULLISH":
            # Use bull call spread for defined risk
            return self.create_bull_call_spread(symbol, current_price, volatility)

        elif market_outlook == "BEARISH":
            # Create bear put spread
            buy_strike = current_price
            sell_strike = current_price * 0.90
            expiration = datetime.now() + timedelta(days=30)
            T = 30 / 365

            if not SCIPY_AVAILABLE:
                return None

            buy_premium = BlackScholes.put_price(
                current_price, buy_strike, T, self.risk_free_rate, volatility
            )
            sell_premium = BlackScholes.put_price(
                current_price, sell_strike, T, self.risk_free_rate, volatility
            )

            legs = [
                OptionLeg(OptionType.PUT, OptionAction.BUY, buy_strike, expiration, 1, buy_premium),
                OptionLeg(OptionType.PUT, OptionAction.SELL, sell_strike, expiration, 1, sell_premium)
            ]

            net_debit = (buy_premium - sell_premium) * 100
            max_risk = net_debit
            max_profit = (buy_strike - sell_strike - (buy_premium - sell_premium)) * 100
            breakeven = buy_strike - (buy_premium - sell_premium)

            return OptionsSignal(
                symbol=symbol,
                strategy_name='bear_put_spread',
                legs=legs,
                confidence=0.75,
                max_risk=max_risk,
                max_profit=max_profit,
                breakeven_points=[breakeven]
            )

        elif market_outlook == "NEUTRAL":
            # Use iron condor
            return self.create_iron_condor(symbol, current_price, volatility)

        return None

    def calculate_greeks(self, leg: OptionLeg, current_price: float,
                        volatility: float) -> Dict[str, float]:
        """Calculate Greeks for an option leg"""
        if not SCIPY_AVAILABLE:
            return {}

        days_to_expiry = (leg.expiration - datetime.now()).days
        T = max(days_to_expiry / 365, 0.001)  # Avoid division by zero

        option_type_str = 'call' if leg.option_type == OptionType.CALL else 'put'

        greeks = {
            'delta': BlackScholes.delta(current_price, leg.strike, T,
                                       self.risk_free_rate, volatility, option_type_str),
            'gamma': BlackScholes.gamma(current_price, leg.strike, T,
                                       self.risk_free_rate, volatility),
            'theta': BlackScholes.theta(current_price, leg.strike, T,
                                       self.risk_free_rate, volatility, option_type_str),
            'vega': BlackScholes.vega(current_price, leg.strike, T,
                                     self.risk_free_rate, volatility)
        }

        return greeks
