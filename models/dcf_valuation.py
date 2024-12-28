# Source Code References
# 1. Discounted Cash Flow (DCF) explained with formula and examples
#    Author: Fernando, J.
#    Date: Sep 20, 2024
#    URL: https://www.investopedia.com/terms/d/dcf.asp
#
# This whole file was made by Alamri with some help of Ai
# I have a lot of knowledge in Value investing and thus whole file is just trying to setup the traders view for
# the longer term as a value investor while also having the technical indicator
# This whole file is just get the fiscal earning and use the fourmlas
# For further explanination please visit the refrence
# Reference: Fernando, J. (2024) [1]

import os
import pickle

import pandas as pd
import yfinance as yf

from data.utils import create_directory

pd.set_option('future.no_silent_downcasting', True)


class DCFValuation:
    def __init__(self, ticker, data_dir: str = 'data/raw_data'):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data_dir = data_dir
        create_directory(self.data_dir)  # Create directory if it does not exist

    def fetch_financial_data(self):
        file_path = os.path.join(self.data_dir, f'{self.ticker}_financial_data.pkl')

        # load from cache
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print("Loaded cached DCF data.")
                return cached_data
            except Exception as e:
                print(f"Error loading cached DCF data: {str(e)}")

        try:
            cf = self.stock.cash_flow
            bs = self.stock.balance_sheet
            is_ = self.stock.income_stmt

            # Save data to cache
            data = (cf, bs, is_)
            try:
                file_path = os.path.join(self.data_dir, f'{self.ticker}_financial_data.pkl')
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                print("Saved DCF data to cache.")
            except Exception as e:
                print(f"Error saving cached DCF data: {str(e)}")

            return cf, bs, is_
        except Exception as e:
            raise Exception(f"Error fetching financial data: {str(e)}")

    def calculate_fcf_historical(self, cf, bs):
        try:
            operating_cash_flow = cf.loc['Operating Cash Flow']
            capex = cf.loc['Capital Expenditure'].abs()
            fcf = operating_cash_flow - capex
            return fcf
        except Exception as e:
            raise Exception(f"Error calculating historical FCF: {str(e)}")

    def calculate_growth_rates(self, fcf_historical):
        growth_rates = fcf_historical.ffill().pct_change()  # Corrected line
        avg_growth_rate = growth_rates.mean()

        max_growth = 0.15
        min_growth = 0.02

        growth_rate = max(min(avg_growth_rate, max_growth), min_growth)
        terminal_growth = 0.02

        return growth_rate, terminal_growth

    def calculate_wacc(self, is_, bs):
        try:
            risk_free_rate = 0.0425
            market_risk_premium = 0.06
            beta = self.stock.info.get('beta', 1.0)
            cost_of_equity = risk_free_rate + beta * market_risk_premium

            interest_expense = abs(is_.loc['Interest Expense'].iloc[0])
            total_debt = bs.loc['Total Debt'].iloc[0]
            if total_debt > 0:
                cost_of_debt = interest_expense / total_debt
            else:
                cost_of_debt = 0

            tax_rate = is_.loc['Tax Rate For Calcs'].iloc[0]

            market_cap = self.stock.info.get('marketCap', 0)
            total_capital = market_cap + total_debt

            if total_capital > 0:
                equity_weight = market_cap / total_capital
                debt_weight = total_debt / total_capital
            else:
                equity_weight = 1
                debt_weight = 0

            wacc = (equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate))

            return max(wacc, 0.08)
        except (KeyError, IndexError) as e:  # Catch KeyError for missing index and IndexError for empty series
            print(f"Error calculating WACC: {str(e)}. Using default WACC of 10%. Check financial data.")
            return 0.1  # Return a default WACC if there's an error

    def project_fcf(self, last_fcf, growth_rate, terminal_growth, periods=5):
        fcf_projections = []
        current_fcf = last_fcf

        for _ in range(periods):
            current_fcf *= (1 + growth_rate)
            fcf_projections.append(current_fcf)

        terminal_value = (fcf_projections[-1] * (1 + terminal_growth) / (self.wacc - terminal_growth))

        return fcf_projections, terminal_value

    def calculate_intrinsic_value(self):
        try:
            cf, bs, is_ = self.fetch_financial_data()

            historical_fcf = self.calculate_fcf_historical(cf, bs)

            growth_rate, terminal_growth = self.calculate_growth_rates(historical_fcf)

            self.wacc = self.calculate_wacc(is_, bs)

            last_fcf = historical_fcf.iloc[0]
            fcf_projections, terminal_value = self.project_fcf(last_fcf, growth_rate, terminal_growth)

            pv_fcf = sum([
                fcf / ((1 + self.wacc) ** (i + 1))
                for i, fcf in enumerate(fcf_projections)
            ])

            pv_terminal = terminal_value / ((1 + self.wacc) ** len(fcf_projections))

            total_value = pv_fcf + pv_terminal

            shares_outstanding = self.stock.info.get('sharesOutstanding', 0)

            if shares_outstanding > 0:
                intrinsic_value = total_value / shares_outstanding
            else:
                raise ValueError("Could not get shares outstanding")

            current_price = self.stock.info.get('currentPrice', 0)

            return {
                'intrinsic_value': intrinsic_value,
                'current_price': current_price,
                'upside': ((intrinsic_value - current_price) / current_price * 100),
                'wacc': self.wacc * 100,
                'growth_rate': growth_rate * 100,
                'terminal_growth': terminal_growth * 100
            }

        except Exception as e:
            raise Exception(f"Error calculating intrinsic value: {str(e)}")
