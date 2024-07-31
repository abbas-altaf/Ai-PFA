import warnings
warnings.filterwarnings('ignore')

import os
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np
from pandasgui import show
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import ta

def get_industries():
    url = "https://stockanalysis.com/stocks/industry/all"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    industries = soup.find_all('table', class_='svelte-qmv8b3')
    industries_list = []
    for industry in industries:
        rows = industry.find_all('tr')
        for row in rows:
            columns = row.find_all('td')
            if columns:
                industry_name = columns[0].text.strip()
                industry_href = columns[0].find('a')['href']
                industries_list.append({'name': industry_name, 'href': industry_href})
    return industries_list

def get_companies(industry):
    url = f'https://stockanalysis.com{industry}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id='main-table')
    companies = []
    if table:
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if cols:
                company_name = cols[2].text.strip()
                symbol_link = cols[1].find('a')
                if symbol_link:
                    symbol = symbol_link.text.strip()
                    companies.append({'ticker': symbol, 'name': company_name})
    return companies

def get_financial_data(company):
    tck = yf.Ticker(company['ticker'])
    income_st = tck.get_income_stmt(pretty=True)
    balance_sh = tck.get_balance_sheet(pretty=True)
    cash_fl = tck.get_cash_flow(pretty=True)

    financial_data = {
        "Income Statement": income_st,
        "Balance Sheet": balance_sh,
        "Cash Flow": cash_fl
    }

    return financial_data

def display_financial_data(financial_data, data_type):
    if data_type in financial_data:
        df = financial_data[data_type]
        show(df)
    else:
        print("Invalid data type selected.")

def get_stock_data(company):
    ticker = company['ticker']
    start = '2014-01-01'
    end = datetime.now().strftime('%Y-%m-%d')
    
    stock = yf.Ticker(ticker).history(interval='1d', start=start, end=end)
    
    return stock

def numeric_fin_analysis(data):
    print(87)
    income_st = data['Income Statement']
    balance_sh = data['Balance Sheet']
    
    ret_on_asset = income_st.loc['Net Income'] / balance_sh.loc['Total Assets']
    ret_on_eq = income_st.loc['Net Income'] / balance_sh.loc['Total Equity Gross Minority Interest']
    nopat = income_st.loc['Operating Income'] - income_st.loc['Tax Provision']
    return_on_inv_cap = nopat / (balance_sh.loc['Total Equity Gross Minority Interest'] + balance_sh.loc['Total Debt'])
    # ret_on_asset = [ret_on_asset]  # Make it a single row
    # print(tabulate([ret_on_asset], headers=['Return on Asset'], tablefmt='pretty'))
    print("Showing return ratios")
    print("Return on Assets")
    print(ret_on_asset)
    # print("Showing return ratios")
    print("Return on Equity")
    print(ret_on_eq)
    print("Return on Invested Capital (ROIC)")
    print(return_on_inv_cap)
    print("\n\n")
    
    print("Showing profit margins over time")
    gr_pr_ma = income_st.loc['Gross Profit'] / income_st.loc['Total Revenue']
    exp_ratio = income_st.loc['Operating Expense'] / income_st.loc['Total Revenue']
    op_pro_ma = income_st.loc['Operating Income'] / income_st.loc['Total Revenue']
    print("gross profit Margin")
    print(gr_pr_ma)
    print("Expense Ratio")
    print(exp_ratio)
    print("Operating Profit Margin")
    print(op_pro_ma)
    
    ebt = income_st.loc['Pretax Income']
    tax_burden = income_st.loc['Net Income'] / ebt
    print("Tax Burden")
    print(tax_burden)
    
    int_burden = ebt / income_st.loc['EBIT']
    print("Interest Burden")
    print(int_burden)
    
    asset_turnover = income_st.loc['Total Revenue'] / balance_sh.loc['Total Assets']
    cash_turnover = income_st.loc['Total Revenue'] / balance_sh.loc['Cash And Cash Equivalents']
    cash_days = (balance_sh.loc['Cash And Cash Equivalents'] * 365) / income_st.loc['Total Revenue']
    account_receivable = income_st.loc['Total Revenue'] / balance_sh.loc['Accounts Receivable']
    account_payable = income_st.loc['Total Revenue'] / balance_sh.loc['Accounts Payable']
    
    print("Asset turnover")
    print(asset_turnover)
    print("Cash turnover")
    print(cash_turnover)
    print("Cash Days")
    print(cash_days)
    print("Accounts Receivable")
    print(account_receivable)
    print("Accounts Payable")
    print(account_payable)
    
    lev = balance_sh.loc['Total Debt'] / balance_sh.loc['Total Equity Gross Minority Interest']
    asset_to_eq = balance_sh.loc['Total Assets'] / balance_sh.loc['Total Equity Gross Minority Interest']
    debt_to_ebtida = balance_sh.loc['Total Debt'] / income_st.loc['EBITDA']
    
    print("Leverage")
    print(lev)
    print("Asset to Equity")
    print(asset_to_eq)
    print("Debt to EBTIDA")
    print(debt_to_ebtida)

def graphical_fin_analysis(data):
    income_st = data['Income Statement']
    balance_sh = data['Balance Sheet']
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # Adjust the number of rows and columns as needed
    
    # Flatten the 2D array of axes for easier iteration
    axs = axs.flatten()
    
    # Plot Return on Assets, Return on Equity, Return on Invested Capital
    ret_on_asset = income_st.loc['Net Income'] / balance_sh.loc['Total Assets']
    ret_on_eq = income_st.loc['Net Income'] / balance_sh.loc['Total Equity Gross Minority Interest']
    nopat = income_st.loc['Operating Income'] - income_st.loc['Tax Provision']
    return_on_inv_cap = nopat / (balance_sh.loc['Total Equity Gross Minority Interest'] + balance_sh.loc['Total Debt'])
    
    axs[0].plot(ret_on_asset.index, ret_on_asset, label='Return on Assets (ROA)', marker='o')
    axs[0].plot(ret_on_eq.index, ret_on_eq, label='Return on Equity (ROE)', marker='o')
    axs[0].plot(return_on_inv_cap.index, return_on_inv_cap, label='Return on Invested Capital (ROIC)', marker='o')
    axs[0].set_title('Return Ratios Over Time')
    axs[0].set_xlabel('Period')
    axs[0].set_ylabel('Ratio Values')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Gross Profit Margin, Expense Ratio, Operating Profit Margin
    gr_pr_ma = income_st.loc['Gross Profit'] / income_st.loc['Total Revenue']
    exp_ratio = income_st.loc['Operating Expense'] / income_st.loc['Total Revenue']
    op_pro_ma = income_st.loc['Operating Income'] / income_st.loc['Total Revenue']

    axs[1].plot(gr_pr_ma.index, gr_pr_ma, label='Gross Profit Margin', marker='o')
    axs[1].plot(exp_ratio.index, exp_ratio, label='Expense Ratio', marker='o')
    axs[1].plot(op_pro_ma.index, op_pro_ma, label='Operating Profit Margin', marker='o')
    axs[1].set_title('Profit Margins Over Time')
    axs[1].set_xlabel('Period')
    axs[1].set_ylabel('Ratio Values')
    axs[1].grid(True)
    axs[1].legend()

    # Plot Tax Burden
    ebt = income_st.loc['Pretax Income']
    tax_burden = income_st.loc['Net Income'] / ebt

    axs[2].plot(tax_burden.index, tax_burden, label='Tax Burden', marker='o')
    axs[2].set_title('Tax Burden Over Time')
    axs[2].set_xlabel('Period')
    axs[2].set_ylabel('Ratio Values')
    axs[2].grid(True)
    axs[2].legend()

    # Plot Interest Burden
    int_burden = ebt / income_st.loc['EBIT']

    axs[3].plot(int_burden.index, int_burden, label='Interest Burden', marker='o')
    axs[3].set_title('Interest Burden Over Time')
    axs[3].set_xlabel('Period')
    axs[3].set_ylabel('Ratio Values')
    axs[3].grid(True)
    axs[3].legend()

    # Plot Turnover Ratios
    asset_turnover = income_st.loc['Total Revenue'] / balance_sh.loc['Total Assets']
    cash_turnover = income_st.loc['Total Revenue'] / balance_sh.loc['Cash And Cash Equivalents']
    cash_days = (balance_sh.loc['Cash And Cash Equivalents'] * 365) / income_st.loc['Total Revenue']
    account_receivable = income_st.loc['Total Revenue'] / balance_sh.loc['Accounts Receivable']
    account_payable = income_st.loc['Total Revenue'] / balance_sh.loc['Accounts Payable']
    # inventory_turn = income_st.loc['Total Revenue'] / balance_sh.loc['Inventory']
    # working_cap = account_receivable + balance_sh.loc['Inventory'] - account_payable

    axs[4].plot(asset_turnover.index, asset_turnover, label='Asset Turnover', marker='o')
    axs[4].plot(cash_turnover.index, cash_turnover, label='Cash Turnover', marker='o')
    axs[4].plot(cash_days.index, cash_days, label='Cash Days', marker='o')
    axs[4].plot(account_receivable.index, account_receivable, label='Account Receivable', marker='o')
    axs[4].plot(account_payable.index, account_payable, label='Account Payable', marker='o')
    # axs[4].plot(inventory_turn.index, inventory_turn, label='Inventory Turnover', marker='o')
    axs[4].set_title('Turnover Ratios Over Time')
    axs[4].set_xlabel('Period')
    axs[4].set_ylabel('Ratio Values')
    axs[4].grid(True)
    axs[4].legend()

    # Plot Leverage Ratios
    lev = balance_sh.loc['Total Debt'] / balance_sh.loc['Total Equity Gross Minority Interest']
    asset_to_eq = balance_sh.loc['Total Assets'] / balance_sh.loc['Total Equity Gross Minority Interest']
    debt_to_ebtida = balance_sh.loc['Total Debt'] / income_st.loc['EBITDA']

    axs[5].plot(lev.index, lev, label='Leverage', marker='o')
    axs[5].plot(asset_to_eq.index, asset_to_eq, label='Assets to Equity', marker='o')
    axs[5].plot(debt_to_ebtida.index, debt_to_ebtida, label='Debt to EBITDA', marker='o')
    axs[5].set_title('Leverage Ratios Over Time')
    axs[5].set_xlabel('Period')
    axs[5].set_ylabel('Ratio Values')
    axs[5].grid(True)
    axs[5].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def linear_regression_forecast(data):
    from sklearn.linear_model import LinearRegression
    def fetch_financial_statement(ticker, statement_code):
        company = yf.Ticker(ticker)
        if statement_code == '1':  # Income Statement
            data = company.financials.T
        elif statement_code == '2':  # Balance Sheet
            data = company.balance_sheet.T
        elif statement_code == '3':  # Cash Flow
            data = company.cashflow.T
        else:
            raise ValueError("Invalid statement type")
        return data

    def preprocess_data(data):
        # Fill NaN values with the mean of each column
        return data.fillna(data.mean())

    def forecast_statement(data, years_to_forecast):
        forecasted_data = pd.DataFrame()
        last_year = data.index[0].year  # Find the last available year in the data
        for column in data.columns:
            X = np.array(range(len(data))).reshape(-1, 1)  # Reshape for sklearn
            y = data[column].fillna(data[column].mean()).values  # Fill NaN values
            model = LinearRegression()
            model.fit(X, y)
            future_years = np.array(range(len(data), len(data) + years_to_forecast)).reshape(-1, 1)
            forecast = model.predict(future_years)
            forecasted_data[column] = forecast
        forecasted_data.index = [last_year + i for i in range(1, years_to_forecast + 1)]
        return forecasted_data
    
    ticker = input("Enter the ticker of the company: ")
    years_to_forecast = int(input("Enter the number of years you want to forecast beyond 2023: "))
    print("Select the financial statement to forecast:\n1: Income Statement\n2: Balance Sheet\n3: Cash Flow")
    statement_code = input("Enter 1, 2, or 3: ")
    
    print(f"Fetching data for {ticker}...")
    data = fetch_financial_statement(ticker, statement_code)
    data = preprocess_data(data)  # Preprocess to handle NaN values
    print(f"Original data (latest 3 years):\n{data.tail(3)}")

    print("Forecasting...")
    forecasted_data = forecast_statement(data, years_to_forecast)
    print("Forecasted Financial Statement for the next {} years:".format(years_to_forecast))
    show(forecasted_data.T)
    
def dcf_model_forecast(data):
    # Placeholder for DCF model forecasting
    def fetch_data(ticker):
        stock = yf.Ticker(ticker)
        financials = stock.financials.T.bfill().ffill()
        balance_sheet = stock.balance_sheet.T.bfill().ffill()
        cash_flows = stock.cashflow.T.bfill().ffill()
        beta = stock.info['beta']
        current_price = stock.info['currentPrice']
        shares_outstanding = stock.info['sharesOutstanding']
        
        capex = -cash_flows.get('Capital Expenditure', pd.Series([0])).iloc[0]
        
        return financials, balance_sheet, cash_flows, stock, beta, current_price, shares_outstanding, capex

    def calculate_wacc(stock, beta):
        risk_free_rate = 0.015  # Reduced risk-free rate to reflect current market conditions
        market_return = 0.08
        equity_cost = risk_free_rate + beta * (market_return - risk_free_rate)
        
        total_debt = stock.balance_sheet.get('Total Debt', pd.Series([0])).iloc[0]
        interest_expense = stock.financials.get('Interest Expense', pd.Series([0])).iloc[0]
        
        if total_debt > 0:
            debt_cost = interest_expense / total_debt
        else:
            debt_cost = 0.03  # Assume a default cost of debt if no debt is reported
        
        total_equity = stock.info.get('marketCap', 1)
        equity_ratio = total_equity / (total_debt + total_equity) if (total_debt + total_equity) != 0 else 0
        debt_ratio = total_debt / (total_debt + total_equity) if (total_debt + total_equity) != 0 else 0
        
        tax_rate = stock.financials.get('Tax Provision', pd.Series([0])).iloc[0] / stock.financials.get('Pretax Income', pd.Series([1])).iloc[0]

        wacc = (equity_ratio * equity_cost) + (debt_ratio * debt_cost * (1 - tax_rate))
        return wacc, equity_cost, debt_cost

    def calculate_dcf(financials, wacc, years, shares_outstanding, capex, ebit_growth_rate=0.12, terminal_growth_rate=0.05):
        ebit = financials.get('Operating Income', pd.Series([0])).iloc[0]
        depreciation = financials.get('Depreciation', pd.Series([ebit * 0.10])).iloc[0]
        delta_nwc = ebit * 0.05

        tax_rate = 0.21
        
        # Adjust EBIT growth
        ebit = [ebit * (1 + ebit_growth_rate) ** year for year in range(years)]
        fcf = [(ebit[year] * (1 - tax_rate) + depreciation - capex - delta_nwc) for year in range(years)]
        
        print("Detailed Financial Metrics:")
        print(f"Initial EBIT: {ebit[0]}, Depreciation: {depreciation}, CapEx: {capex}, ΔNWC: {delta_nwc}, Tax Rate: {tax_rate}")
        print(f"Free Cash Flow for each year: {fcf}")
        
        pv_of_fcf = np.sum([fcf[year] / ((1 + wacc) ** (year + 1)) for year in range(years)])
        
        terminal_value = fcf[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
        terminal_value_pv = terminal_value / ((1 + wacc) ** years)

        npv = pv_of_fcf + terminal_value_pv
        intrinsic_value_per_share = npv / shares_outstanding
        
        return intrinsic_value_per_share
    
    ticker = input("Enter the ticker symbol for the company: ")
    years = int(input("Enter the number of years forward for DCF forecasting: "))
    
    financials, balance_sheet, cash_flows, stock, beta, current_price, shares_outstanding, capex = fetch_data(ticker)
    wacc, equity_cost, debt_cost = calculate_wacc(stock, beta)
    intrinsic_value_per_share = calculate_dcf(financials, wacc, years, shares_outstanding, capex)
    
    print(f"Current Share Price: ${current_price:.2f}")
    print(f"Intrinsic Share Price after {years} years: ${intrinsic_value_per_share:.2f}")
    print(f"WACC: {wacc:.2%}, Cost of Equity: {equity_cost:.2%}, Cost of Debt: {debt_cost:.2%}, Beta: {beta}")
    
    recommendation = 'Buy' if intrinsic_value_per_share > current_price else 'Sell' if intrinsic_value_per_share < current_price else 'Hold'
    print(f"Recommendation: {recommendation}")


def select_data_type(financial_data):
    while True:
        print("Select financial data type:")
        print("1. Income Statement")
        print("2. Balance Sheet")
        print("3. Cash Flow")
        print("4. Go Back")
        
        financial_data_choice = int(input("Enter the number of your choice: "))
        financial_data_types = ["Income Statement", "Balance Sheet", "Cash Flow"]
        
        if financial_data_choice == 4:
            break
        elif 1 <= financial_data_choice <= 3:
            selected_financial_data_type = financial_data_types[financial_data_choice - 1]
            display_financial_data(financial_data, selected_financial_data_type)
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def select_financial_analysis(data):
    while True:
        print("Select analysis type:")
        print("1. Numeric Analysis")
        print("2. Graphical Analysis")
        print("3. Go Back")
        
        analysis_choice = int(input("Enter the number of your choice: "))
        
        if analysis_choice == 1:
            numeric_fin_analysis(data)
        elif analysis_choice == 2:
            graphical_fin_analysis(data)
        elif analysis_choice == 3:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

def select_financial_forecasting(data):
    while True:
        print("Select forecasting method:")
        print("1. Linear Regression")
        print("2. DCF Model")
        print("3. Go Back")
        
        forecasting_choice = int(input("Enter the number of your choice: "))
        
        if forecasting_choice == 1:
            linear_regression_forecast(data)
        elif forecasting_choice == 2:
            dcf_model_forecast(data)
        elif forecasting_choice == 3:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

def select_financial_comparison():
# Function to download financial statement data and other necessary metrics
    def download_financials(tickers):
        financials = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            financials[ticker] = {
                'info': stock.info,
                'income': stock.financials.T,
                'balance': stock.balance_sheet.T,
                'cashflow': stock.cashflow.T
            }
        return financials

    # Function to plot and compare a financial metric
    def compare_financials(financials, tickers, title, calculation_function, is_higher_better, description):
        plt.figure(figsize=(14, 7))
        best_value = float('-inf') if is_higher_better else float('inf')
        best_ticker = None
        best_year = None
        plotted = False

        for ticker in tickers:
            value = calculation_function(financials[ticker], ticker)
            if value is not None and not value.empty:
                plt.plot(value.index, value, label=f'{ticker} {title}')
                plotted = True
                latest_value = value.dropna().iloc[0]
                latest_year = value.dropna().index[0].year
                print(latest_year)
                print(latest_value)
                if (is_higher_better and latest_value > best_value) or (not is_higher_better and latest_value < best_value):
                    best_value = latest_value
                    best_ticker = ticker
                    best_year = latest_year

        if plotted:
            plt.title(f"{title} Comparison")
            plt.xlabel("Year")
            plt.ylabel(title)
            plt.legend()
            plt.show()
            print(description)
            comparison_type = "higher" if is_higher_better else "lower"
            print(f"Best performer for {title}: {best_ticker} with {title} of {best_value:.2f} in {best_year} ({comparison_type} is better)")
        else:
            plt.close()
            print(f"No data available to plot for {title}.")

    # Mapping choices to metrics and calculation functions
    def get_total_revenue(data, ticker):
        return data['income'].get('Total Revenue')

    def get_net_income(data, ticker):
        return data['income'].get('Net Income')

    def get_operating_profit_margin(data, ticker):
        revenue = data['income'].get('Total Revenue')
        operating_income = data['income'].get('EBIT')
        if revenue is not None and operating_income is not None:
            return operating_income / revenue

    # Liquidity Ratios
    def get_current_ratio(data, ticker):
        total_current_assets = data['balance'].get('Total Current Assets')
        total_current_liabilities = data['balance'].get('Total Current Liabilities')
        if total_current_assets is not None and total_current_liabilities is not None:
            return total_current_assets / total_current_liabilities

    def get_quick_ratio(data, ticker):
        total_current_assets = data['balance'].get('Total Current Assets')
        inventory = data['balance'].get('Inventory')
        total_current_liabilities = data['balance'].get('Total Current Liabilities')
        if total_current_assets is not None and inventory is not None and total_current_liabilities is not None:
            return (total_current_assets - inventory) / total_current_liabilities

    def get_cash_ratio(data, ticker):
        cash_and_equivalents = data['balance'].get('Cash And Cash Equivalents')
        total_current_liabilities = data['balance'].get('Total Current Liabilities')
        if cash_and_equivalents is not None and total_current_liabilities is not None:
            return cash_and_equivalents / total_current_liabilities

    # Solvency Ratios
    def get_debt_to_equity(data, ticker):
        total_liabilities = data['balance'].get('Total Liabilities Net Minority Interest')
        total_equity = data['balance'].get('Common Stock Equity')
        if total_liabilities is not None and total_equity is not None and not total_equity.empty:
            if total_equity.iloc[-1] != 0:  # Avoid division by zero
                return total_liabilities / total_equity
            else:
                print(f"{ticker}: Equity is zero, cannot compute Debt to Equity ratio.")
        else:
            print(f"{ticker}: Data for Total Liabilities or Total Equity is not available.")
        return None

    def get_debt_ratio(data, ticker):
        total_liabilities = data['balance'].get('Total Liabilities Net Minority Interest')
        total_assets = data['balance'].get('Total Assets')
        if total_liabilities is not None and total_assets is not None:
            return total_liabilities / total_assets

    def get_interest_coverage_ratio(data, ticker):
        ebit = data['income'].get('EBIT')
        interest_expense = data['income'].get('Interest Expense')
        if ebit is not None and interest_expense is not None:
            return ebit / interest_expense

    # Profitability Ratios
    def get_roa(data, ticker):
        net_income = data['income'].get('Net Income')
        total_assets = data['balance'].get('Total Assets')
        if net_income is not None and total_assets is not None:
            return net_income / total_assets

    def get_roe(data, ticker):
        net_income = data['income'].get('Net Income')
        total_equity = data['balance'].get('Common Stock Equity')
        if net_income is not None and total_equity is not None:
            return net_income / total_equity

    def get_roi(data, ticker):
        net_income = data['income'].get('Net Income')
        total_liabilities = data['balance'].get('Total Liabilities Net Minority Interest')
        total_equity = data['balance'].get('Common Stock Equity')
        if net_income is not None and total_liabilities is not None and total_equity is not None:
            return net_income / (total_liabilities + total_equity)

    def get_pe_ratio(data, ticker):
        return pd.Series(data['info'].get('forwardPE'), index=[pd.to_datetime('now')])

    def get_long_term_debt(data, ticker):
        return data['balance'].get('Long Term Debt')

    def get_enterprise_value(data, ticker):
        ev = data['info'].get('enterpriseValue')
        if ev is not None:
            return pd.Series([ev], index=[pd.to_datetime('now')])

    def get_free_cash_flow(data, ticker):
        operating_cash_flow = data['cashflow'].get('Total Cash From Operating Activities')
        capital_expenditures = data['cashflow'].get('Capital Expenditures')
        if operating_cash_flow is not None and capital_expenditures is not None:
            return operating_cash_flow + capital_expenditures

    metric_functions = {
        1: {"func": get_total_revenue, "is_higher_better": True, "description": "Total Revenue indicates the total income generated by the company from its business activities."},
        2: {"func": get_net_income, "is_higher_better": True, "description": "Net Income shows the company's profit after all expenses have been deducted from total revenue."},
        3: {"func": get_operating_profit_margin, "is_higher_better": True, "description": "Operating Profit Margin measures the percentage of revenue left after paying for variable costs of production."},
        4: {"func": get_current_ratio, "is_higher_better": True, "description": "Current Ratio measures a company's ability to pay short-term obligations."},
        5: {"func": get_quick_ratio, "is_higher_better": True, "description": "Quick Ratio measures a company's ability to meet its short-term obligations with its most liquid assets."},
        6: {"func": get_cash_ratio, "is_higher_better": True, "description": "Cash Ratio measures a company's ability to pay off short-term debt with cash and cash equivalents."},
        7: {"func": get_debt_to_equity, "is_higher_better": False, "description": "Debt to Equity Ratio shows the proportion of equity and debt the company uses to finance its assets. A lower ratio is better."},
        8: {"func": get_debt_ratio, "is_higher_better": False, "description": "Debt Ratio measures the proportion of a company's assets that are financed by debt. A lower ratio is better."},
        9: {"func": get_interest_coverage_ratio, "is_higher_better": True, "description": "Interest Coverage Ratio measures how easily a company can pay interest on its outstanding debt. A higher ratio is better."},
        10: {"func": get_roa, "is_higher_better": True, "description": "Return on Assets (ROA) measures how efficiently a company can manage its assets to produce profits."},
        11: {"func": get_roe, "is_higher_better": True, "description": "Return on Equity (ROE) measures the profitability of a business in relation to the equity."},
        12: {"func": get_roi, "is_higher_better": True, "description": "Return on Investment (ROI) measures the gain or loss generated on an investment relative to the amount of money invested."},
        13: {"func": get_pe_ratio, "is_higher_better": False, "description": "P/E Ratio (Price to Earnings) measures a company's current share price relative to its per-share earnings. A lower ratio is generally better."},
        14: {"func": get_long_term_debt, "is_higher_better": False, "description": "Long Term Debt indicates the total amount of long-term financial obligations."},
        15: {"func": get_enterprise_value, "is_higher_better": True, "description": "Enterprise Value (EV) is a measure of a company's total value, often used as a comprehensive alternative to equity market capitalization."},
        16: {"func": get_free_cash_flow, "is_higher_better": True, "description": "Free Cash Flow measures the cash generated by the company that is available for distribution among all the securities holders of a corporate entity."}
    }

    # User input and execution logic
    print("Available Financial Comparisons:")
    for i in range(1, 17):
        print(f"{i}: {metric_functions[i]['func'].__name__[4:].replace('_', ' ')}")
    choice = int(input("Select the comparison type by number: "))
    num_companies = int(input("How many companies do you want to analyze? "))
    tickers = [input(f"Enter ticker {i+1}: ") for i in range(num_companies)]

    financial_data = download_financials(tickers)
    selected_metric = metric_functions[choice]
    compare_financials(financial_data, tickers, selected_metric['func'].__name__[4:].replace('_', ' '), selected_metric['func'], selected_metric['is_higher_better'], selected_metric['description'])

def graph_num_stock_analysis():
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Function to calculate different types of returns
    def calculate_returns(df):
        df['Relative Return'] = df['Close'].pct_change()
        df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Difference Return'] = df['Close'].diff()

    # Function to calculate moving averages
    def calculate_moving_averages(df, window_sizes=[20, 50, 100]):
        for window in window_sizes:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

    # Function to calculate RSI
    def calculate_rsi(df, window=14):
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # Function to calculate MACD
    def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
        df['MACD'] = df['Close'].ewm(span=short_window, adjust=False).mean() - df['Close'].ewm(span=long_window, adjust=False).mean()
        df['MACD Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

    # Function to plot trends
    def plot_trends(df, ticker):
        plt.figure(figsize=(14, 14))

        plt.subplot(3, 2, 1)
        plt.plot(df['Close'], label='Close Price')
        plt.plot(df['MA_20'], label='MA 20')
        plt.plot(df['MA_50'], label='MA 50')
        plt.plot(df['MA_100'], label='MA 100')
        plt.title(f'{ticker} Close Price and Moving Averages')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(df['Open'], label='Open Price')
        plt.plot(df['High'], label='High Price')
        plt.plot(df['Low'], label='Low Price')
        plt.plot(df['Close'], label='Close Price')
        plt.title(f'{ticker} Stock Prices (Open, High, Low, Close)')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(df['Relative Return'], label='Relative Return')
        plt.title(f'{ticker} Simple Return')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(df['Log Return'], label='Log Return')
        plt.title(f'{ticker} Log Return')
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(df['Difference Return'], label='Difference Return')
        plt.title(f'{ticker} Difference Return')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(df['RSI'], label='RSI')
        plt.title(f'{ticker} Relative Strength Index (RSI)')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(df['MACD'], label='MACD')
        plt.plot(df['MACD Signal'], label='MACD Signal')
        plt.title(f'{ticker} MACD')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Function to assess company performance
    def assess_performance(df):
        recent_close = df['Close'].iloc[-1]
        ma_50 = df['MA_50'].iloc[-1]
        ma_100 = df['MA_100'].iloc[-1]

        if recent_close > ma_50 > ma_100:
            return f"The company is performing well. The recent closing price (${recent_close:.2f}) is above both the 50-day (${ma_50:.2f}) and the 100-day (${ma_100:.2f}) moving averages, indicating a positive trend."
        else:
            return f"The company is not performing well. The recent closing price (${recent_close:.2f}) is below the 50-day (${ma_50:.2f}) and/or the 100-day (${ma_100:.2f}) moving averages, indicating a negative trend."

    # Main function to get data and perform analysis
    def analyze_stock(ticker):
        # Download data
        df = yf.download(ticker, period='1y')
        
        # Calculate returns
        calculate_returns(df)
        
        # Calculate moving averages
        calculate_moving_averages(df)
        
        # Calculate RSI
        calculate_rsi(df)
        
        # Calculate MACD
        calculate_macd(df)
        
        # Plot trends
        plot_trends(df, ticker)
        
        # Assess performance
        performance = assess_performance(df)
        print(performance)
        
    ticker = input("Enter the ticker symbol of the company: ")
    analyze_stock(ticker)

def select_stock_analysis(data):
    while True:
        print("Select analysis type:")
        print("1. Graphical and Numerical Analysis")
        print("3. Go Back")
        
        analysis_choice = int(input("Enter the number of your choice: "))
        
        if analysis_choice == 1:
            graph_num_stock_analysis()
        elif analysis_choice == 3:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

def select_stock_forecasting(data):
    while True:
        print("Select forecasting method:")
        print("1. Training Testing (Models) & Prediction")
        print("2. Go Back")
        
        forecasting_choice = int(input("Enter the number of your choice: "))
        
        if forecasting_choice == 1:
            train_testing_stock(data)
        elif forecasting_choice == 2:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 2.")
            
def train_testing_stock(data):
    print("Calculating MSE of GRU.....")
    gru_mse,gru_train_dates,gru_test_dates,gru_scaler_target,gru_y_train,gru_actual_prices,gru_predicted_prices = mse_gru(data)
    print("Calculating MSE of LSTM.....")
    lstm_mse,lstm_train_dates,lstm_test_dates,lstm_scaler_target,lstm_y_train,lstm_actual_prices,lstm_predicted_prices=mse_lstm(data)
    print("Calculating MSE of Random Forest.....")
    rf_mse,data,rf_X_test,rf_ensemble_predictions=mse_random_forest(data)
    
    if gru_mse < lstm_mse and gru_mse < rf_mse:
        print("The best model for this company is GRU. The mean square error of GRU is ",gru_mse)
        plot_gru(gru_train_dates,gru_test_dates,gru_scaler_target,gru_y_train,gru_actual_prices,gru_predicted_prices)
        print("The next day prices of the model GRU are the following: ")
        one_day_gru(data)
    if lstm_mse < gru_mse and lstm_mse < rf_mse:
        print("The best model for this company is LSTM. The mean square error of LSTM is ",lstm_mse)
        plot_lstm(lstm_train_dates,lstm_test_dates,lstm_scaler_target,lstm_y_train,lstm_actual_prices,lstm_predicted_prices)
        print("The next day prices of the model LSTM are the following: ")
        one_day_lstm(data)
    if rf_mse < lstm_mse and rf_mse < gru_mse:
        print("The best model for this company is Random Forest. The mean square error of Random Forest is ",rf_mse)
        plot_random_forest(data,rf_X_test,rf_ensemble_predictions)
        print("The next day prices of the model Random Forest are the following: ")
        one_day_rand_for(data)

def mse_gru(data):
    data.reset_index(inplace=True, drop=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature Engineering: Adding more technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['BB_high'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_hband()
    data['BB_low'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_lband()

    data.dropna(inplace=True)  # Drop rows with NaN values

    features = data[['High', 'Low', 'Close', 'MA50', 'MA200', 'RSI', 'BB_high', 'BB_low']]
    target = data['Open']

    # Scaling data
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

    # Creating sequences for GRU
    def create_sequences(features, target, time_steps=1):
        X, y = [], []
        for i in range(len(features) - time_steps):
            X.append(features[i:(i + time_steps)])
            y.append(target[i + time_steps])
        return np.array(X), np.array(y)

    time_steps = 20  # Increased time steps for longer sequences
    X, y = create_sequences(scaled_features, scaled_target, time_steps)

    # Splitting data based on dates
    train_end_date = '2022-12-31'
    train_idx = data.index <= train_end_date
    test_idx = data.index > train_end_date

    X_train = X[train_idx[time_steps:]]
    y_train = y[train_idx[time_steps:]]
    X_test = X[test_idx[time_steps:]]
    y_test = y[test_idx[time_steps:]]

    # Dates for plotting
    dates = data.index[time_steps:]
    train_dates = dates[train_idx[time_steps:]]
    test_dates = dates[test_idx[time_steps:]]

    # GRU Model
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        GRU(50),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Adding EarlyStopping and ModelCheckpoint
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5.keras', save_best_only=True, monitor='val_loss')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1,
            callbacks=[early_stop, model_checkpoint])

    # Prediction and evaluation
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler_target.inverse_transform(predicted_prices)
    actual_prices = scaler_target.inverse_transform(y_test)

    # Calculate MSE
    mse = mean_squared_error(actual_prices, predicted_prices)

    return mse,train_dates,test_dates,scaler_target,y_train,actual_prices,predicted_prices

def mse_lstm(data):
    data.reset_index(inplace=True, drop=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature Engineering: Adding more technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['BB_high'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_hband()
    data['BB_low'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_lband()

    data.dropna(inplace=True)  # Drop rows with NaN values

    features = data[['High', 'Low', 'Close', 'MA50', 'MA200', 'RSI', 'BB_high', 'BB_low']]
    target = data['Open']

    # Scaling data
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

    # Creating sequences for LSTM
    def create_sequences(features, target, time_steps=1):
        X, y = [], []
        for i in range(len(features) - time_steps):
            X.append(features[i:(i + time_steps)])
            y.append(target[i + time_steps])
        return np.array(X), np.array(y)

    time_steps = 20  # Increased time steps for longer sequences
    X, y = create_sequences(scaled_features, scaled_target, time_steps)

    # Splitting data based on dates
    train_end_date = '2022-12-31'
    train_idx = data.index <= train_end_date
    test_idx = data.index > train_end_date

    X_train = X[train_idx[time_steps:]]
    y_train = y[train_idx[time_steps:]]
    X_test = X[test_idx[time_steps:]]
    y_test = y[test_idx[time_steps:]]

    # Dates for plotting
    dates = data.index[time_steps:]
    train_dates = dates[train_idx[time_steps:]]
    test_dates = dates[test_idx[time_steps:]]

    # Simplified LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Adding EarlyStopping and ModelCheckpoint
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5.keras', save_best_only=True, monitor='val_loss')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1,
            callbacks=[early_stop, model_checkpoint])

    # Prediction and evaluation
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler_target.inverse_transform(predicted_prices)
    actual_prices = scaler_target.inverse_transform(y_test)

    # Calculate MSE
    mse = mean_squared_error(actual_prices, predicted_prices)
    # print(f"Mean Squared Error: {mse}")

    return mse,train_dates,test_dates,scaler_target,y_train,actual_prices,predicted_prices

def mse_random_forest(data):
    data.reset_index(inplace=True, drop=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.dayofweek
    data['Prev_Close'] = data['Close'].shift(1)
    data['Change'] = data['Close'] - data['Prev_Close']
    data['Volatility'] = (data['High'] - data['Low']) / data['Low']
    data['Rolling_Mean'] = data['Close'].rolling(window=5).mean()
    data['Rolling_Std'] = data['Close'].rolling(window=5).std()
    data['Rolling_Mean_20'] = data['Close'].rolling(window=20).mean()  # New feature
    data['Rolling_Std_20'] = data['Close'].rolling(window=20).std()   # New feature
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()  # Exponential Moving Average
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)
    data['Lag_3'] = data['Close'].shift(3)
    data.dropna(inplace=True)  # Drop rows with NaN values
    data.set_index('Date', inplace=True)

    # Selecting features and target
    features = data[['Year', 'Month', 'Day', 'High', 'Low', 'Close', 'Prev_Close', 'Change', 'Volatility', 'Rolling_Mean', 'Rolling_Std', 'Rolling_Mean_20', 'Rolling_Std_20', 'EMA_20', 'Lag_1', 'Lag_2', 'Lag_3']]
    target = data['Open']

    # Splitting the data
    X_train = features[features['Year'] <= 2021]
    X_test = features[features['Year'] >= 2022]
    y_train = target.loc[X_train.index]
    y_test = target.loc[X_test.index]

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Adding polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)  # Increased splits

    # Optimized hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=100, cv=tscv, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train_poly, y_train)
    best_model_rf = random_search.best_estimator_

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train_poly, y_train)

    # Ensemble of Random Forest and Gradient Boosting
    predictions_rf = best_model_rf.predict(X_test_poly)
    predictions_gbr = gbr.predict(X_test_poly)
    ensemble_predictions = (predictions_rf + predictions_gbr) / 2

    # Evaluating the ensemble model
    mse = mean_squared_error(y_test, ensemble_predictions)
    return mse,data,X_test,ensemble_predictions

def plot_gru(train_dates,test_dates,scaler_target,y_train,actual_prices,predicted_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(train_dates, scaler_target.inverse_transform(y_train), label='Training Actual Prices')
    plt.plot(test_dates, actual_prices, label='Testing Actual Prices')
    plt.plot(test_dates, predicted_prices, label='Predicted Prices', linestyle='--')
    plt.title('GRU Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Opening Price')
    plt.legend()
    plt.show()

def plot_lstm(train_dates,test_dates,scaler_target,y_train,actual_prices,predicted_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(train_dates, scaler_target.inverse_transform(y_train), label='Training Actual Prices')
    plt.plot(test_dates, actual_prices, label='Testing Actual Prices')
    plt.plot(test_dates, predicted_prices, label='Predicted Prices', linestyle='--')
    plt.title('LSTM Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Opening Price')
    plt.legend()
    plt.show()

def plot_random_forest(data,X_test,ensemble_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Open'], label='Actual Prices')
    plt.plot(X_test.index, ensemble_predictions, label='Predicted Prices', linestyle='--')
    plt.title('Random Forest Stock Price Prediction')
    plt.xlabel('Year')
    plt.ylabel('Stock Opening Price')
    plt.legend()
    plt.show()

def one_day_gru(data):
    today = datetime.now()
    tomorrow = (today + timedelta(days=1)).strftime('%Y-%m-%d')  # Calculates tomorrow's date

    data.reset_index(inplace=True, drop=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature Engineering: Adding more technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['BB_high'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_hband()
    data['BB_low'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_lband()

    data.dropna(inplace=True)  # Drop rows with NaN values

    features = data[['High', 'Low', 'Close', 'MA50', 'MA200', 'RSI', 'BB_high', 'BB_low']]
    target = data[['Open', 'High', 'Low', 'Close']]  # Multivariate target

    # Scaling data
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target)

    # Creating sequences for GRU
    def create_sequences(features, target, time_steps=1):
        X, y = [], []
        for i in range(len(features) - time_steps):
            X.append(features[i:(i + time_steps)])
            y.append(target[i + time_steps])
        return np.array(X), np.array(y)

    time_steps = 20  # Increased time steps for longer sequences
    X, y = create_sequences(scaled_features, scaled_target, time_steps)

    # Splitting data based on dates
    train_end_date = '2022-12-31'
    train_idx = data.index <= train_end_date
    test_idx = data.index > train_end_date

    X_train = X[train_idx[time_steps:]]
    y_train = y[train_idx[time_steps:]]
    X_test = X[test_idx[time_steps:]]
    y_test = y[test_idx[time_steps:]]

    # Multivariate GRU model
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        GRU(50),
        Dropout(0.3),
        Dense(4)  # Output layer for 4 target values
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Adding EarlyStopping and ModelCheckpoint
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5.keras', save_best_only=True, monitor='val_loss')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1,
            callbacks=[early_stop, model_checkpoint])

    # Prediction and evaluation
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler_target.inverse_transform(predicted_prices)
    actual_prices = scaler_target.inverse_transform(y_test)

    # Calculate MSE
    # mse = mean_squared_error(actual_prices, predicted_prices)
    # print(f"Mean Squared Error: {mse}")

    # Predict tomorrow's prices
    last_sequence = scaled_features[-time_steps:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    tomorrow_prediction = model.predict(last_sequence)
    tomorrow_prediction = scaler_target.inverse_transform(tomorrow_prediction)

    print(f"Tomorrow's price prediction for {tomorrow} through GRU is:\n"
        f"Open: {tomorrow_prediction[0, 0]:.2f}\n"
        f"High: {tomorrow_prediction[0, 1]:.2f}\n"
        f"Low: {tomorrow_prediction[0, 2]:.2f}\n"
        f"Close: {tomorrow_prediction[0, 3]:.2f}")
    
def one_day_lstm(data):
    today = datetime.now()
    tomorrow = (today + timedelta(days=1)).strftime('%Y-%m-%d')  # Calculates tomorrow's date
    
    data.reset_index(inplace=True, drop=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature Engineering: Adding more technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['BB_high'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_hband()
    data['BB_low'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_lband()

    data.dropna(inplace=True)  # Drop rows with NaN values

    features = data[['High', 'Low', 'Close', 'MA50', 'MA200', 'RSI', 'BB_high', 'BB_low']]
    target = data[['Open', 'High', 'Low', 'Close']]  # Multivariate target

    # Scaling data
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target)

    # Creating sequences for LSTM
    def create_sequences(features, target, time_steps=1):
        X, y = [], []
        for i in range(len(features) - time_steps):
            X.append(features[i:(i + time_steps)])
            y.append(target[i + time_steps])
        return np.array(X), np.array(y)

    time_steps = 20  # Increased time steps for longer sequences
    X, y = create_sequences(scaled_features, scaled_target, time_steps)

    # Splitting data based on dates
    train_end_date = '2022-12-31'
    train_idx = data.index <= train_end_date
    test_idx = data.index > train_end_date

    X_train = X[train_idx[time_steps:]]
    y_train = y[train_idx[time_steps:]]
    X_test = X[test_idx[time_steps:]]
    y_test = y[test_idx[time_steps:]]

    # Multivariate LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(4)  # Output layer for 4 target values
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Adding EarlyStopping and ModelCheckpoint
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5.keras', save_best_only=True, monitor='val_loss')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1,
            callbacks=[early_stop, model_checkpoint])

    # Prediction and evaluation
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler_target.inverse_transform(predicted_prices)
    actual_prices = scaler_target.inverse_transform(y_test)

    # Calculate MSE
    # mse = mean_squared_error(actual_prices, predicted_prices)
    # print(f"Mean Squared Error: {mse}")

    # Predict tomorrow's prices
    last_sequence = scaled_features[-time_steps:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    tomorrow_prediction = model.predict(last_sequence)
    tomorrow_prediction = scaler_target.inverse_transform(tomorrow_prediction)

    print(f"Tomorrow's price prediction for {tomorrow} through LSTM is:\n"
        f"Open: {tomorrow_prediction[0, 0]:.2f}\n"
        f"High: {tomorrow_prediction[0, 1]:.2f}\n"
        f"Low: {tomorrow_prediction[0, 2]:.2f}\n"
        f"Close: {tomorrow_prediction[0, 3]:.2f}")
    
def one_day_rand_for(data):
    today = datetime.now()
    tomorrow = (today + timedelta(days=1)).strftime('%Y-%m-%d')  # Calculates tomorrow's date
    
    data.reset_index(inplace=True, drop=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature Engineering
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['BB_high'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_hband()
    data['BB_low'] = ta.volatility.BollingerBands(close=data['Close'], window=20).bollinger_lband()
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()

    data.dropna(inplace=True)

    features = data[['High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'BB_high', 'BB_low', 'EMA10', 'ATR']]
    target = data[['Open', 'High', 'Low', 'Close']]

    # Scaling data
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target)

    # Creating sequences for features
    def create_sequences(features, target, time_steps=1):
        X, y = [], []
        for i in range(time_steps, len(features)):
            X.append(features[i - time_steps:i].flatten())
            y.append(target[i])
        return np.array(X), np.array(y)

    time_steps = 40  # Use past 40 days to predict next day
    X, y = create_sequences(scaled_features, scaled_target, time_steps)

    # Splitting data based on dates
    train_end_date = '2022-12-31'
    train_idx = data.index <= train_end_date
    test_idx = data.index > train_end_date

    X_train = X[train_idx[time_steps:]]
    y_train = y[train_idx[time_steps:]]
    X_test = X[test_idx[time_steps:]]
    y_test = y[test_idx[time_steps:]]

    # Enhanced Random Forest Model
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # Prediction and evaluation
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler_target.inverse_transform(predicted_prices)
    actual_prices = scaler_target.inverse_transform(y_test)

    # Calculate MSE
    # mse = mean_squared_error(actual_prices, predicted_prices)
    # print(f"Mean Squared Error: {mse}")

    # Predict tomorrow's prices
    last_sequence = scaled_features[-time_steps:]
    last_sequence = last_sequence.flatten().reshape(1, -1)  # Reshape for single prediction
    tomorrow_prediction = model.predict(last_sequence)
    tomorrow_prediction = scaler_target.inverse_transform(tomorrow_prediction)

    print(f"Tomorrow's price prediction for {tomorrow} through Random Forest is:\n"
        f"Open: {tomorrow_prediction[0, 0]:.2f}\n"
        f"High: {tomorrow_prediction[0, 1]:.2f}\n"
        f"Low: {tomorrow_prediction[0, 2]:.2f}\n"
        f"Close: {tomorrow_prediction[0, 3]:.2f}")
    
def select_stock_portfolio(data):
    while True:
        print("Select method:")
        print("1. Markowitz Model")
        print("2. Monte Carlo")
        print("3. Go Back")
        
        port_choice = int(input("Enter the number of your choice: "))
        
        if port_choice == 1:
            markowitz_portf()
        if port_choice == 2:
            monte_carlo_port()
        elif port_choice == 3:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 2.")
        
def markowitz_portf():
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import datetime
    from cvxopt import matrix, solvers

    def get_top_tickers(num_tickers):
        """Fetch tickers manually defined."""
        tickers = ["AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "JNJ", "V", "UNH"]
        return tickers[:num_tickers]

    def get_stock_data(tickers):
        """Fetch stock historical data."""
        end_date = datetime.datetime.today()
        start_date = end_date - datetime.timedelta(days=365)
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data = data.dropna(how='all').ffill().bfill()
        return data

    def calculate_returns(prices):
        """Calculate daily returns from price data."""
        returns = prices.pct_change().dropna()
        return returns

    def calculate_portfolio_statistics(returns, weights):
        """Calculate expected return, volatility, and Sharpe ratio."""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        expected_return = np.dot(weights, mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = expected_return / volatility
        return expected_return, volatility, sharpe_ratio

    def optimize_portfolio(returns, investment):
        """Optimize the portfolio to balance return and risk."""
        num_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        P = matrix(cov_matrix.values)  # Minimize variance
        q = matrix(np.zeros(num_assets))  # No linear component
        G = matrix(-np.eye(num_assets))  # Non-negativity constraint
        h = matrix(np.zeros(num_assets))
        A = matrix(1.0, (1, num_assets))  # Sum of weights = 1
        b = matrix(1.0)

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        weights = np.array(sol['x']).flatten()

        expected_return, volatility, sharpe_ratio = calculate_portfolio_statistics(returns, weights)
        investment_weights = {returns.columns[i]: (100 * weights[i], weights[i] * investment) for i in range(num_assets)}
        return expected_return, volatility, sharpe_ratio, investment_weights
    
    try:
        num_companies = int(input("Enter the number of companies you want to invest in: "))
        investment_amount = float(input("Enter your total investment amount: $"))
        top_tickers = get_top_tickers(num_companies)

        print("Attempting to fetch data for tickers:", top_tickers)
        prices = get_stock_data(top_tickers)
        returns = calculate_returns(prices)
        exp_return, volatility, sharpe_ratio, weights = optimize_portfolio(returns, investment_amount)
        
        print("\nOptimized Portfolio:")
        print(f"Weights (Percentage, Dollar Amount): {weights}")
        print(f"Expected Annual Return: {exp_return:.2%}")
        print(f"Risk (Volatility): {volatility:.2%}")
        print(f"Dollar Return: ${exp_return * investment_amount:.2f}")
        print(f"Dollar Risk: ${volatility * investment_amount:.2f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


    
def monte_carlo_port():
    print("Monte Carlo Portfolio ")
    def get_top_tickers(num_tickers):
        tickers = ["AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "JNJ", "V", "UNH"]
        return tickers[:num_tickers]

    def get_stock_data(tickers):
        """Fetch stock historical data and ensure data is aligned, handling failed downloads."""
        end_date = datetime.today()
        start_date = end_date - datetime.timedelta(days=365)
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        if isinstance(data, pd.DataFrame):
            data = data.dropna(how='all')
            data = data.ffill().bfill()
        else:
            # If the result is not a DataFrame, it means no data was returned
            raise ValueError("Failed to download any data.")
        return data

    def calculate_returns(prices):
        """Calculate daily returns from price data, handling NA values manually."""
        returns = prices.pct_change(fill_method=None).dropna()
        if returns.empty:
            raise ValueError("No sufficient data to compute returns.")
        return returns

    def calculate_portfolio_statistics(returns, weights):
        """Calculate expected return, volatility, and Sharpe ratio."""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        expected_return = np.dot(weights, mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = expected_return / volatility
        return expected_return, volatility, sharpe_ratio

    def optimize_portfolio(returns, investment, risk_preference='moderate'):
        num_assets = len(returns.columns)
        num_portfolios = 10000
        all_weights = np.zeros((num_portfolios, num_assets))
        results = np.zeros((3, num_portfolios))

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            all_weights[i, :] = weights  # Store the weights in the matrix
            expected_return, volatility, sharpe_ratio = calculate_portfolio_statistics(returns, weights)
            results[:, i] = expected_return, volatility, sharpe_ratio

        if risk_preference == 'risk_taker':
            index = np.argmax(results[0, :])  # Maximize return
        elif risk_preference == 'risk_averse':
            index = np.argmin(results[1, :])  # Minimize volatility
        else:  # Moderate risk preference
            index = np.argmax(results[2, :])  # Maximize Sharpe ratio

        optimal_weights = all_weights[index, :]
        optimal_return, optimal_volatility, optimal_sharpe = results[:, index]

        investment_weights = {returns.columns[i]: (100 * optimal_weights[i], optimal_weights[i] * investment)
                            for i in range(num_assets) if optimal_weights[i] > 0.01}  # Filter small weights for clarity

        return optimal_return, optimal_volatility, optimal_sharpe, investment_weights
    
    num_companies = int(input("Enter the number of companies you want to invest in: "))
    investment_amount = float(input("Enter your total investment amount: $"))
    top_tickers = get_top_tickers(num_companies)

    print("Attempting to fetch data for tickers:", top_tickers)
    prices = get_stock_data(top_tickers)
    returns = calculate_returns(prices)

    for risk_preference in ['risk_taker', 'risk_averse', 'moderate']:
        exp_return, volatility, sharpe_ratio, weights = optimize_portfolio(returns, investment_amount, risk_preference)
        print(f"\n{risk_preference.capitalize()} Portfolio:")
        print(f"Weights (Percentage, Dollar Amount): {weights}")
        print(f"Expected Annual Return: {exp_return:.2%}, Risk (Volatility): {volatility:.2%}")
        print(f"Dollar Return: ${exp_return * investment_amount:.2f}, Dollar Risk: ${volatility * investment_amount:.2f}")

        
def select_stocks_comparison():
    #Comparison of STOCKS
    import yfinance as yf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import mplfinance as mpf

    # Function to download stock data
    def download_data(tickers, period="max", interval="1d"):
        data = yf.download(tickers, period=period, interval=interval)
        return data

    # Function to plot and compare returns
    def compare_returns(data):
        close = data['Close']
        
        # Calculating returns
        difference_returns = close.diff()
        relative_returns = close.pct_change()
        log_returns = np.log(close / close.shift(1))

        # Debug: Print last few entries to verify distinct values
        print("Sample Difference Returns:", difference_returns.tail())
        print("Sample Relative Returns:", relative_returns.tail())
        print("Sample Log Returns:", log_returns.tail())

        return_types = ['Difference', 'Relative', 'Log']
        return_data = [difference_returns, relative_returns, log_returns]

        for return_type, returns in zip(return_types, return_data):
            plt.figure(figsize=(14, 7))
            for ticker in returns.columns:
                plt.plot(returns.index, returns[ticker], label=f'{ticker} {return_type} Return')
            plt.title(f"{return_type} Returns (Full History)")
            plt.legend()
            plt.show()

            # Filter data for the last 60 days
            end_date = returns.index.max()
            start_date = end_date - pd.Timedelta(days=60)
            last_60_days_returns = returns.loc[start_date:end_date]

            average_return_values = last_60_days_returns.mean()
            best_performer = average_return_values.idxmax()
            best_performance_value = average_return_values.max()
            print("Returns: Analyzing returns focuses on the gain or loss of an investment over a period. This comparison tells you how much an investment has grown or shrunk, typically expressed as a percentage. It's fundamental in assessing the performance of a stock or portfolio relative to others or the market, helping investors understand profitability and historical performance.")
            print(f"Best {return_type} Return among tickers (Last 60 Days): {best_performer} with {best_performance_value:.4f}")

    # EMA plotting and comparison function
    def compare_ema(data, tickers):
        best_stability = float('inf')
        best_ticker = None
        for ticker in tickers:
            ema = data['Close'][ticker].ewm(span=20, adjust=False).mean()
            plt.figure(figsize=(14, 7))
            plt.plot(data['Close'][ticker], label='Closing Price')
            plt.plot(ema, label='20-Day EMA')
            plt.title(f"EMA for {ticker} (Full History)")
            plt.legend()
            plt.show()

            # Calculating the stability measure as the average absolute difference
            stability = (data['Close'][ticker] - ema).abs().mean()
            if stability < best_stability:
                best_stability = stability
                best_ticker = ticker
        print("EMA (Exponential Moving Average): The Exponential Moving Average is a type of moving average that places a greater weight and significance on the most recent data points. EMA is particularly useful for identifying trends by smoothing out price data and allowing traders to see market sentiment at a glance. It can be a crucial signal for timing entries and exits in trading strategies.")        
        print(f"Best EMA performer based on stability: {best_ticker} with a stability measure of {best_stability:.4f}")

    # Volume comparison function
    def compare_volume(data, tickers):
        highest_volume = 0
        volume_ticker = None
        for ticker in tickers:
            plt.figure(figsize=(14, 7))
            plt.bar(data.index, data['Volume'][ticker], color='blue')
            plt.title(f"Volume for {ticker} (Full History)")
            plt.show()

            # Finding the ticker with the highest average volume
            avg_volume = data['Volume'][ticker].mean()
            if avg_volume > highest_volume:
                highest_volume = avg_volume
                volume_ticker = ticker
        print("Volume: Volume refers to the number of shares or contracts traded in a security or market during a given period. Volume is a measure of market activity and liquidity; high volume often corresponds to high interest and activity in a stock. Analyzing volume can help confirm trends, as trends with higher volume can be considered more robust and likely to continue.")        
        print(f"Ticker with the highest average volume: {volume_ticker} with an average volume of {highest_volume:.2f}")

    # Bollinger Bands plotting and comparison function
    def compare_bollinger_bands(data, tickers):
        min_band_distance = float('inf')
        band_ticker = None
        for ticker in tickers:
            sma = data['Close'][ticker].rolling(window=20).mean()
            std = data['Close'][ticker].rolling(window=20).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            plt.figure(figsize=(14, 7))
            plt.plot(data['Close'][ticker], label='Closing Price')
            plt.plot(upper_band, label='Upper Bollinger Band')
            plt.plot(lower_band, label='Lower Bollinger Band')
            plt.title(f"Bollinger Bands for {ticker} (Full History)")
            plt.legend()
            plt.show()

            # Measure of how often the price stays within the bands
            in_band = ((data['Close'][ticker] >= lower_band) & (data['Close'][ticker] <= upper_band)).mean()
            if in_band < min_band_distance:
                min_band_distance = in_band
                band_ticker = ticker
        print("Bollinger Bands: Bollinger Bands are a technical analysis tool defined by a set of lines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a stock's price. They help measure volatility and identify overbought or oversold conditions. When prices move outside the bands, it suggests a continuation of the trend, while prices moving inside can signal a potential reversal.")
        print(f"Ticker with best adherence to Bollinger Bands: {band_ticker} with adherence rate of {min_band_distance:.4f}")

    # MACD plotting and comparison function
    def compare_macd(data, tickers):
        best_macd = -float('inf')
        macd_ticker = None
        for ticker in tickers:
            exp1 = data['Close'][ticker].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'][ticker].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            plt.figure(figsize=(14, 7))
            plt.plot(data.index, macd, label='MACD')
            plt.plot(data.index, signal, label='Signal Line')
            plt.title(f"MACD for {ticker} (Full History)")
            plt.legend()
            plt.show()

            # Determining the best bullish signal
            bullish_signal = (macd - signal).mean()
            if bullish_signal > best_macd:
                best_macd = bullish_signal
                macd_ticker = ticker
        print("MACD (Moving Average Convergence Divergence): The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. It consists of the MACD line (the difference between two EMAs), a signal line (an EMA of the MACD line), and a histogram showing the difference between the MACD line and the signal line. MACD helps identify trend direction, momentum, and potential reversals.")
        print(f"Ticker with best MACD performance: {macd_ticker} with an average divergence of {best_macd:.4f}")

    def compare_candlestick(data, tickers):
        best_stability = float('inf')
        best_ticker = None
        for ticker in tickers:
            # Extract data for the current ticker
            if isinstance(data.columns, pd.MultiIndex):
                ticker_data = data.xs(ticker, level=1, axis=1)
            else:
                ticker_data = data  # If already specific to one ticker

            # Ensure that the DataFrame has a DatetimeIndex
            if not isinstance(ticker_data.index, pd.DatetimeIndex):
                ticker_data.index = pd.to_datetime(ticker_data.index)

            # Ensure columns are correctly named as expected by mplfinance
            if not set(['Open', 'High', 'Low', 'Close']).issubset(ticker_data.columns):
                raise ValueError("Data must include 'Open', 'High', 'Low', 'Close' columns")

            # Setting up the plot for candlestick
            plt.figure(figsize=(14, 7))
            mpf.plot(ticker_data, type='candle', style='charles', title=f"Candlestick chart for {ticker} (Full History)")
            plt.show()

            # Calculating the simple moving average (SMA) for stability calculation
            sma = ticker_data['Close'].rolling(window=20).mean()
            
            # Calculating the stability measure as the average absolute difference from the SMA
            stability = (ticker_data['Close'] - sma).abs().mean()
            if stability < best_stability:
                best_stability = stability
                best_ticker = ticker

        print("Candlestick Patterns: Candlestick patterns are a method of charting used to describe price movements of a security, derivative, or currency. They can provide insight into market sentiment and potential price movements. Recognizing these patterns can help traders predict short-term price movements based on historical patterns, aiding in making trading decisions on when to enter or exit trades.")
        print(f"Best performer based on price stability relative to SMA: {best_ticker} with a stability measure of {best_stability:.4f}")


    # User input and execution logic
    print("Available Analyses:\n1: Returns\n2: EMA\n3: Volume\n4: Bollinger Bands\n5: MACD\n6: Candlestick Patterns")
    analysis_choice = int(input("Select the analysis type by number: "))
    num_companies = int(input("How many companies do you want to analyze? "))
    tickers = [input(f"Enter ticker {i+1}: ") for i in range(num_companies)]

    data = download_data(tickers)
    if analysis_choice == 1:
        compare_returns(data)
    elif analysis_choice == 2:
        compare_ema(data, tickers)
    elif analysis_choice == 3:
        compare_volume(data, tickers)
    elif analysis_choice == 4:
        compare_bollinger_bands(data, tickers)
    elif analysis_choice == 5:
        compare_macd(data, tickers)
    elif analysis_choice == 6:
        compare_candlestick(data, tickers)


def select_company(industry_href, industry_name):
    companies = get_companies(industry_href)
    clear_screen()
    items_per_page = 10
    page = 0

    while True:
        start_index = page * items_per_page
        end_index = start_index + items_per_page
        current_page_companies = companies[start_index:end_index]

        print(f"Select a company in {industry_name}:")
        for i, company in enumerate(current_page_companies, start=start_index + 1):
            print(f"{i}. {company['ticker']}, {company['name']}")

        # Check if there are more companies to show or provide an option to go back
        menu_options = []
        if end_index < len(companies):
            menu_options.append(f"{end_index + 1}. Show more")
        menu_options.append(f"{len(menu_options) + end_index + 1}. Go Back")
        
        for option in menu_options:
            print(option)

        try:
            company_choice = int(input("Enter the number of your choice: "))
            clear_screen()

            if company_choice == end_index + 1 and "Show more" in menu_options[0]:
                page += 1  # Move to the next page of companies
                continue
            elif company_choice == len(menu_options) + end_index:
                break  # Go back to the previous menu
            elif start_index < company_choice <= end_index and company_choice <= len(companies):
                selected_company = companies[company_choice - 1]
                print(f"You selected {selected_company['ticker']} - {selected_company['name']}.")
                # Further processing can be added here
            else:
                raise ValueError
        except ValueError:
            # clear_screen()
            print("900 Invalid choice. Please enter a valid number.\n")
            
        print("What data would you like to view?")
        print("1. Financial Statements")
        print("2. Stocks")
        print("3. Go Back")
        
        data_choice = int(input("Enter the number of your choice: "))
        
        if data_choice == 1:
            financial_data = get_financial_data(selected_company)
            while True:
                print("What would you like to do?")
                print("1. Show Data")
                print("2. Analysis")
                print("3. Forecasting")
                print("4. Comparison")
                print("5. Go Back")
                
                action_choice = int(input("Enter the number of your choice: "))
                
                if action_choice == 1:
                    select_data_type(financial_data)
                elif action_choice == 2:
                    select_financial_analysis(financial_data)
                elif action_choice == 3:
                    select_financial_forecasting(financial_data)
                elif action_choice == 4:
                    select_financial_comparison()
                elif action_choice == 5:
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
        elif data_choice == 2:
            stock_data = get_stock_data(selected_company)
            while True:
                print("What would you like to do?")
                print("1. Show Data")
                print("2. Analysis")
                print("3. Forecasting")
                print("4. Portfolio")
                print("5. Comparison")
                print("6. Go Back")
                
                action_choice = int(input("Enter the number of your choice: "))
                
                if action_choice == 1:
                    show(stock_data)
                elif action_choice == 2:
                    select_stock_analysis(stock_data)
                elif action_choice == 3:
                    select_stock_forecasting(stock_data)
                elif action_choice == 4:
                    select_stock_portfolio(stock_data)
                elif action_choice == 5:
                    select_stocks_comparison()
                elif action_choice == 6:
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 5.")
        elif data_choice == 3:
            continue
        else:
            print("Invalid choice.")

def clear_screen():
    os.system('cls')
    
def main():
    clear_screen()
    print("=============")
    print("MY Personal Finance")
    print("=============\n")

    industries = get_industries()
    items_per_page = 50
    page = 0

    while True:
        start_index = page * items_per_page
        end_index = start_index + items_per_page
        current_page_industries = industries[start_index:end_index]
        
        print("Showing list of Industries ...\n")
        for i, industry in enumerate(current_page_industries, start=start_index + 1):
            print(f"{i}. {industry['name']}")
        
        # Check if there are more industries to show
        if end_index < len(industries):
            print(f"{end_index + 1}. Show more")
        else:
            print(f"{end_index + 1}. Exit\n")
        
        try:
            industry_choice = int(input("Enter the number of your choice: "))
            clear_screen()

            if industry_choice == end_index + 1:
                if end_index < len(industries):
                    page += 1  # Move to the next page of industries
                    continue
                else:
                    print("Exiting program.")
                    break
            elif start_index < industry_choice <= end_index and industry_choice <= len(industries):
                selected_industry = industries[industry_choice - 1]
                print(f"You selected {selected_industry['name']}.")
                select_company(selected_industry['href'], selected_industry['name'])
            else:
                raise ValueError
        except ValueError:
            # clear_screen()
            print("Invalid choice. Please enter a valid number.\n")

if __name__ == "__main__":
    main()