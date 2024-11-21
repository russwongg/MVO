#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import xlsxwriter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import mplcursors

from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import KFold


# In[7]:


import subprocess
import sys

# List of required libraries
required_libraries = [
    'numpy', 
    'pandas', 
    'matplotlib', 
    'scipy',
    'tkinter',
    'datetime',
    'xlsxwriter',
    'scikit-learn'
]

# Function to install missing libraries
def install_libraries():
    for library in required_libraries:
        try:
            __import__(library)  # Try importing the library
        except ImportError:
            print(f"{library} is not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Call the function at the beginning of the script
install_libraries()


# In[3]:


matplotlib.use('TkAgg')  # Use TkAgg backend for embedding in Tkinter

class MVOApp:
    def __init__(self, master):
        self.master = master
        master.title("Mean-Variance Optimizer with Backtesting")

        # Set initial window size
        master.geometry('1200x800')

        # Initialize variables
        self.data = None
        self.fund_names = []
        self.asset_classes = {}  # Mapping fund names to asset classes
        self.constraints = {}
        self.entries = {}
        self.result_text = tk.StringVar()
        self.optimize_choice = tk.StringVar(value='Maximize Sharpe Ratio')
        self.simulation_count = 0
        self.simulations = []
        self.efficient_frontier_data = None
        self.fund_stats = {}  # For fund returns and volatilities

        self.selected_simulation = None  # For storing the selected simulation

        # Model information for popups
        self.model_info = {
            "Basic MVO": {
                "Strengths": "Simple, foundational, flexible.",
                "WhenToUse": "When you have reliable data and need a straightforward model."
            },
            "Ledoit-Wolf Shrinkage": {
                "Strengths": "Robust covariance estimates, reduces estimation error.",
                "WhenToUse": "Use with small datasets or many assets."
            },
            "K-Fold Cross-Validation": {
                "Strengths": "Reduces overfitting, better generalization.",
                "WhenToUse": "To enhance robustness and avoid overfitting."
            },
            "Combined Ledoit-Wolf and K-Fold": {
                "Strengths": "Best estimation accuracy, robust and generalizable portfolios.",
                "WhenToUse": "With small samples and high estimation risk."
            },
        }
        self.last_model_choice = None  # To track the last selected model

        # Configure the main window to expand widgets properly
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)

        # Create main canvas with scrollbar
        self.canvas = tk.Canvas(master)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.main_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.main_frame, anchor='nw')

        self.main_frame.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Create UI components
        self.create_widgets()

        # Initialize last_model_choice to prevent popup at startup
        self.last_model_choice = self.model_choice.get()

    def _on_mousewheel(self, event):
        # Windows and MacOS
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self):
        # File selection button
        self.file_button = tk.Button(self.main_frame, text="Choose your file...", command=self.load_file)
        self.file_button.pack(pady=5)

        # Data read status and sample data display (used as log now)
        self.data_status = tk.StringVar()
        self.data_status_label = tk.Label(self.main_frame, textvariable=self.data_status, fg='green')
        self.data_status_label.pack()
        self.sample_data_text = tk.Text(self.main_frame, height=10, width=80)
        self.sample_data_text.pack()

        # Frame for fund constraints and asset allocation constraints
        self.constraints_allocation_frame = tk.Frame(self.main_frame)
        self.constraints_allocation_frame.pack(pady=5)

        # Frame for fund constraints
        self.constraints_frame = tk.Frame(self.constraints_allocation_frame)
        self.constraints_frame.pack(side=tk.LEFT, padx=5)

        # Frame for asset allocation constraints
        self.allocation_frame = tk.Frame(self.constraints_allocation_frame)
        self.allocation_frame.pack(side=tk.LEFT, padx=5)

        # Asset Allocation Constraints
        tk.Label(self.allocation_frame, text="Asset Allocation Constraints").grid(row=0, column=0, columnspan=2, pady=5)
        self.asset_classes_list = ["Equities", "Fixed Income", "Alternatives"]
        self.allocation_entries = {}
        total_row = len(self.asset_classes_list) + 1
        for i, asset_class in enumerate(self.asset_classes_list):
            tk.Label(self.allocation_frame, text=f"{asset_class} Allocation (%)").grid(row=i+1, column=0, sticky='e', padx=5, pady=2)
            entry = tk.Entry(self.allocation_frame, width=10)
            entry.grid(row=i+1, column=1, padx=5, pady=2)
            self.allocation_entries[asset_class] = entry

        # Add a label to show total allocation
        tk.Label(self.allocation_frame, text="Total Allocation (%)").grid(row=total_row, column=0, sticky='e', padx=5, pady=2)
        self.total_allocation_label = tk.Label(self.allocation_frame, text="0%")
        self.total_allocation_label.grid(row=total_row, column=1, padx=5, pady=2)

        # Bind events to allocation entries to update total allocation
        for entry in self.allocation_entries.values():
            entry.bind("<KeyRelease>", self.update_total_allocation)

        # Now, move the date selection, optimization goal, and model selection below the list of funds
        # Date selection dropdowns
        self.date_frame = tk.Frame(self.main_frame)
        self.date_frame.pack(pady=5)
        tk.Label(self.date_frame, text="Start date:").pack(side=tk.LEFT)
        self.start_date_var = tk.StringVar()
        self.start_date_dropdown = ttk.Combobox(self.date_frame, textvariable=self.start_date_var, width=10, state='readonly')
        self.start_date_dropdown.pack(side=tk.LEFT, padx=5)

        tk.Label(self.date_frame, text="End date:").pack(side=tk.LEFT)
        self.end_date_var = tk.StringVar()
        self.end_date_dropdown = ttk.Combobox(self.date_frame, textvariable=self.end_date_var, width=10, state='readonly')
        self.end_date_dropdown.pack(side=tk.LEFT, padx=5)

        # Bind events to date dropdowns to update fund stats when dates change
        self.start_date_dropdown.bind("<<ComboboxSelected>>", self.update_fund_stats)
        self.end_date_dropdown.bind("<<ComboboxSelected>>", self.update_fund_stats)

        # Optimization choice
        self.optimize_frame = tk.Frame(self.main_frame)
        self.optimize_frame.pack(pady=5)
        tk.Label(self.optimize_frame, text="Optimization Goal:").pack(side=tk.LEFT)
        self.optimize_options = ["Maximize Returns", "Maximize Sharpe Ratio", "Minimize Volatility"]
        for option in self.optimize_options:
            tk.Radiobutton(self.optimize_frame, text=option, variable=self.optimize_choice, value=option).pack(side=tk.LEFT)

        # Optimization Model Selection
        self.model_frame = tk.Frame(self.main_frame)
        self.model_frame.pack(pady=5)
        tk.Label(self.model_frame, text="Select Optimization Model:").pack(side=tk.LEFT)
        self.model_choice = tk.StringVar(value="Basic MVO")
        self.model_options = ["Basic MVO", "Ledoit-Wolf Shrinkage", "K-Fold Cross-Validation", "Combined Ledoit-Wolf and K-Fold"]
        self.model_dropdown = ttk.Combobox(self.model_frame, textvariable=self.model_choice, values=self.model_options, state='readonly')
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        self.model_choice.trace('w', self.update_folds_entry)

        # Number of Folds Entry
        self.folds_frame = tk.Frame(self.main_frame)
        self.folds_frame.pack(pady=5)
        tk.Label(self.folds_frame, text="Number of Folds (3-10):").pack(side=tk.LEFT)
        self.num_folds_var = tk.IntVar(value=5)
        self.num_folds_entry = tk.Spinbox(self.folds_frame, from_=3, to=10, textvariable=self.num_folds_var, width=5)
        self.num_folds_entry.pack(side=tk.LEFT, padx=5)
        self.num_folds_entry.config(state='disabled')  # Initially disabled

        # Additional Constraints Frame
        self.additional_constraints_frame = tk.Frame(self.main_frame)
        self.additional_constraints_frame.pack(pady=5)
        # Minimum Monthly Return Entry
        tk.Label(self.additional_constraints_frame, text="Minimum Monthly Return (%):").pack(side=tk.LEFT)
        self.min_return_var = tk.StringVar()
        self.min_return_entry = tk.Entry(self.additional_constraints_frame, textvariable=self.min_return_var, width=10)
        self.min_return_entry.pack(side=tk.LEFT, padx=5)
        # Maximum Drawdown Entry
        tk.Label(self.additional_constraints_frame, text="Maximum Drawdown (%):").pack(side=tk.LEFT)
        self.max_drawdown_var = tk.StringVar()
        self.max_drawdown_entry = tk.Entry(self.additional_constraints_frame, textvariable=self.max_drawdown_var, width=10)
        self.max_drawdown_entry.pack(side=tk.LEFT, padx=5)

        # Buttons
        self.buttons_frame = tk.Frame(self.main_frame)
        self.buttons_frame.pack(pady=5)

        self.generate_button = tk.Button(self.buttons_frame, text="Generate Results", command=self.generate_results)
        self.generate_button.pack(side=tk.LEFT, padx=2)

        self.clear_constraints_button = tk.Button(self.buttons_frame, text="Clear all constraints", command=self.clear_constraints)
        self.clear_constraints_button.pack(side=tk.LEFT, padx=2)

        self.clear_outputs_button = tk.Button(self.buttons_frame, text="Clear all OUTPUTS", command=self.clear_all_outputs)
        self.clear_outputs_button.pack(side=tk.LEFT, padx=2)
        
        self.all_cap_button = tk.Button(self.buttons_frame, text="All Cap at 10%", command=self.all_cap_10)
        self.all_cap_button.pack(side=tk.LEFT, padx=2)
        
        self.all_cap_button = tk.Button(self.buttons_frame, text="All Cap at 18%", command=self.all_cap_18)
        self.all_cap_button.pack(side=tk.LEFT, padx=2)
        
        self.all_cap_button = tk.Button(self.buttons_frame, text="All Cap at 20%", command=self.all_cap_20)
        self.all_cap_button.pack(side=tk.LEFT, padx=2)

        self.all_cap_25_button = tk.Button(self.buttons_frame, text="All Cap at 25%", command=self.all_cap_25)
        self.all_cap_25_button.pack(side=tk.LEFT, padx=2)

        self.efficient_frontier_button = tk.Button(self.buttons_frame, text="Generate Efficient Frontier", command=self.generate_efficient_frontier)
        self.efficient_frontier_button.pack(side=tk.LEFT, padx=2)

        self.export_button = tk.Button(self.buttons_frame, text="Export Results to Excel", command=self.export_results)
        self.export_button.pack(side=tk.LEFT, padx=2)

        self.backtest_button = tk.Button(self.buttons_frame, text="Backtest", command=self.open_backtest_window)
        self.backtest_button.pack(side=tk.LEFT, padx=2)

        # Results display with scrollbars
        self.results_frame = tk.Frame(self.main_frame, width=900, height=600)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_frame.pack_propagate(0)

        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add vertical scrollbar
        self.results_vscrollbar = tk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.results_vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add horizontal scrollbar
        self.results_hscrollbar = tk.Scrollbar(self.results_frame, orient="horizontal", command=self.results_canvas.xview)
        self.results_hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.results_canvas.configure(yscrollcommand=self.results_vscrollbar.set, xscrollcommand=self.results_hscrollbar.set)

        self.results_inner_frame = tk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_inner_frame, anchor='nw')
        self.results_inner_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )

        # Mouse wheel scrolling for the results canvas
        self.results_canvas.bind_all("<MouseWheel>", self._on_mousewheel_results)
        self.results_canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel_results)

        # Efficient frontier display
        self.chart_frame = tk.Frame(self.main_frame, width=900, height=600)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        self.chart_frame.pack_propagate(0)

    def update_folds_entry(self, *args):
        model = self.model_choice.get()
        if "K-Fold" in model:
            self.num_folds_entry.config(state='normal')
        else:
            self.num_folds_entry.config(state='disabled')

        # Show popup with strengths and when to use
        if model != self.last_model_choice:
            strengths = self.model_info[model]["Strengths"]
            when_to_use = self.model_info[model]["WhenToUse"]
            message = f"{model}\n\nStrengths:\n{strengths}\n\nWhen to use:\n{when_to_use}"
            tk.messagebox.showinfo("Model Information", message)
            self.last_model_choice = model

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xls;*.xlsx")])
        if file_path:
            try:
                # Read Excel file
                # The first row is the header, second row is asset classes
                df = pd.read_excel(file_path, header=None)
                self.fund_names = df.iloc[0, 1:].tolist()
                asset_classes_row = df.iloc[1, 1:].tolist()
                self.asset_classes = dict(zip(self.fund_names, asset_classes_row))
                self.data = df.iloc[2:, :]
                self.data.columns = ['Date'] + self.fund_names
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data.set_index('Date', inplace=True)

                # Convert data to numeric, coerce errors
                self.data = self.data.apply(pd.to_numeric, errors='coerce')

                self.create_constraints_entries()
                # Display data read status and sample data
                self.data_status.set("Data fully read.")
                self.sample_data_text.delete('1.0', tk.END)
                self.sample_data_text.insert(tk.END, "Sample Data (First 5 Rows):\n")
                self.sample_data_text.insert(tk.END, self.data.head().to_string())

                # Populate date dropdowns
                dates = self.data.index.strftime('%b-%y').tolist()
                self.start_date_dropdown['values'] = dates
                self.end_date_dropdown['values'] = dates

                # Set default selection to earliest and latest dates
                if dates:
                    self.start_date_var.set(dates[0])
                    self.end_date_var.set(dates[-1])

                # Initialize fund statistics
                self.update_fund_stats()

            except Exception as e:
                self.data_status.set(f"Failed to read data: {e}")
                self.sample_data_text.delete('1.0', tk.END)
                print(f"Exception during file loading: {e}")

    def create_constraints_entries(self):
        # Clear previous entries
        for widget in self.constraints_frame.winfo_children():
            widget.destroy()
        self.entries.clear()
        self.fund_stats.clear()

        # Add headers
        tk.Label(self.constraints_frame, text="Fund").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        tk.Label(self.constraints_frame, text="Asset Class").grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.constraints_frame, text="Max Weight (%)").grid(row=0, column=2, padx=5, pady=5)
        tk.Label(self.constraints_frame, text="Monthly Return (%)").grid(row=0, column=3, padx=5, pady=5)
        tk.Label(self.constraints_frame, text="Monthly Volatility (%)").grid(row=0, column=4, padx=5, pady=5)

        for i, fund in enumerate(self.fund_names):
            tk.Label(self.constraints_frame, text=fund).grid(row=i+1, column=0, padx=5, pady=2, sticky='w')
            asset_class = self.asset_classes.get(fund, "N/A")
            tk.Label(self.constraints_frame, text=asset_class).grid(row=i+1, column=1, padx=5, pady=2)
            entry = tk.Entry(self.constraints_frame, width=10)
            entry.grid(row=i+1, column=2, padx=5, pady=2)
            self.entries[fund] = entry

            # Create labels for returns and volatilities
            return_label = tk.Label(self.constraints_frame, text="0.00%")
            return_label.grid(row=i+1, column=3, padx=5, pady=2)
            volatility_label = tk.Label(self.constraints_frame, text="0.00%")
            volatility_label.grid(row=i+1, column=4, padx=5, pady=2)

            # Store references to the labels and initialize stats
            self.fund_stats[fund] = {
                'ReturnLabel': return_label,
                'VolatilityLabel': volatility_label,
                'Return': 0.0,
                'Volatility': 0.0
            }

    def update_total_allocation(self, event=None):
        total = 0.0
        for entry in self.allocation_entries.values():
            value = entry.get()
            if value:
                try:
                    total += float(value)
                except ValueError:
                    pass  # Ignore invalid inputs
        self.total_allocation_label.config(text=f"{total:.2f}%")

    def update_fund_stats(self, event=None):
        if self.data is None:
            return

        # Get selected date range
        start_date_str = self.start_date_var.get()
        end_date_str = self.end_date_var.get()

        if start_date_str and end_date_str:
            try:
                # Parse dates
                start_date = datetime.strptime(start_date_str, '%b-%y')
                end_date = datetime.strptime(end_date_str, '%b-%y')

                if start_date > end_date:
                    tk.messagebox.showinfo("Invalid Date Range", "Start date must be before End date.")
                    return

                # Filter data
                data_filtered = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]

                if data_filtered.empty:
                    tk.messagebox.showinfo("No Data", "No data available for the selected date range.")
                    return

                # Recalculate returns and volatilities
                returns = data_filtered.mean()
                volatilities = data_filtered.std()

                for fund in self.fund_names:
                    fund_return = returns[fund] * 100
                    fund_volatility = volatilities[fund] * 100
                    self.fund_stats[fund]['Return'] = fund_return
                    self.fund_stats[fund]['Volatility'] = fund_volatility
                    self.fund_stats[fund]['ReturnLabel'].config(text=f"{fund_return:.2f}%")
                    self.fund_stats[fund]['VolatilityLabel'].config(text=f"{fund_volatility:.2f}%")
            except Exception as e:
                tk.messagebox.showinfo("Error", f"An error occurred: {e}")
        else:
            # If dates are not selected, reset stats to zero
            for fund in self.fund_names:
                self.fund_stats[fund]['Return'] = 0.0
                self.fund_stats[fund]['Volatility'] = 0.0
                self.fund_stats[fund]['ReturnLabel'].config(text="0.00%")
                self.fund_stats[fund]['VolatilityLabel'].config(text="0.00%")

    def clear_constraints(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)

        # Clear additional constraints
        self.min_return_entry.delete(0, tk.END)
        self.max_drawdown_entry.delete(0, tk.END)

        # Clear asset allocation constraints
        for entry in self.allocation_entries.values():
            entry.delete(0, tk.END)
        self.update_total_allocation()

    def all_cap_10(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, '10')

    def all_cap_18(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, '18')
            
    def all_cap_20(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, '20')

    def all_cap_25(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, '25')

    def generate_results(self):
        if self.data is None:
            self.result_text.set("Please load a data file first.")
            return

        # Clear the sample data text (used as log)
        self.sample_data_text.delete('1.0', tk.END)

        # Read fund constraints
        bounds = []
        entered_constraints = {}
        for fund in self.fund_names:
            max_weight = self.entries[fund].get()
            if max_weight:
                try:
                    max_weight_value = float(max_weight) / 100.0
                except ValueError:
                    self.result_text.set(f"Invalid input for {fund}.")
                    return
                bounds.append((0, max_weight_value))
                entered_constraints[fund] = max_weight_value  # Store as fraction
            else:
                max_weight_value = 1.0  # No constraint
                bounds.append((0, max_weight_value))

        # Read asset allocation constraints
        asset_allocation_constraints = {}
        total_allocation = 0.0
        for asset_class in self.asset_classes_list:
            allocation = self.allocation_entries[asset_class].get()
            if allocation:
                try:
                    allocation_value = float(allocation) / 100.0
                    asset_allocation_constraints[asset_class] = allocation_value
                    total_allocation += allocation_value
                except ValueError:
                    self.result_text.set(f"Invalid input for {asset_class} allocation.")
                    return

        if asset_allocation_constraints and abs(total_allocation - 1.0) > 0.01:
            tk.messagebox.showinfo("Invalid Allocation", "Total asset allocation must sum up to 100%.")
            return

        # Ensure weights sum to 1
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Asset allocation constraints
        if asset_allocation_constraints:
            for asset_class, allocation in asset_allocation_constraints.items():
                indices = [i for i, fund in enumerate(self.fund_names) if self.asset_classes[fund] == asset_class]
                if indices:
                    def asset_class_constraint(x, indices=indices, allocation=allocation):
                        return allocation - np.sum([x[i] for i in indices])
                    cons.append({'type': 'eq', 'fun': asset_class_constraint})
                else:
                    tk.messagebox.showinfo("Allocation Error", f"No funds found for asset class {asset_class}.")
                    return

        # Select data based on date range
        start_date_str = self.start_date_var.get()
        end_date_str = self.end_date_var.get()

        if start_date_str and end_date_str:
            try:
                # Parse dates
                start_date = datetime.strptime(start_date_str, '%b-%y')
                end_date = datetime.strptime(end_date_str, '%b-%y')

                if start_date > end_date:
                    self.result_text.set("Start date must be before End date.")
                    return

                # Filter data
                self.data_filtered = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
            except Exception as e:
                self.result_text.set(f"Invalid date selection: {e}")
                return
        else:
            self.data_filtered = self.data.copy()

        if self.data_filtered.empty:
            self.result_text.set("No data available for the selected date range.")
            return

        # Log the steps
        self.sample_data_text.insert(tk.END, f"Starting optimization using {self.model_choice.get()} model.\n")

        # Initialize variables
        x0 = np.array([1.0 / len(self.fund_names)] * len(self.fund_names))

        # Parse additional constraints
        # Minimum monthly return constraint
        min_return_input = self.min_return_var.get()
        if min_return_input:
            try:
                min_return = float(min_return_input) / 100.0
            except ValueError:
                tk.messagebox.showinfo("Invalid Input", "Minimum Monthly Return must be a number.")
                return
            # Add constraint
            avg_returns = self.data_filtered.mean()
            cons.append({'type': 'ineq', 'fun': lambda x: np.dot(x, avg_returns.values) - min_return})
        else:
            min_return = None

        # Maximum drawdown constraint
        max_drawdown_input = self.max_drawdown_var.get()
        if max_drawdown_input:
            try:
                max_drawdown = float(max_drawdown_input) / 100.0  # Convert to fraction
            except ValueError:
                tk.messagebox.showinfo("Invalid Input", "Maximum Drawdown must be a number.")
                return
            # Add constraint
            def max_drawdown_constraint(x):
                portfolio_returns = self.data_filtered.dot(x)
                mdd = self.calculate_max_drawdown(portfolio_returns)
                return max_drawdown - mdd
            cons.append({'type': 'ineq', 'fun': max_drawdown_constraint})
        else:
            max_drawdown = None

        # Initialize variables to avoid UnboundLocalError
        weights = None
        portfolio_return = None
        portfolio_volatility = None
        sharpe_ratio = None
        max_drawdown_value = None

        # Optimization process
        if self.model_choice.get() == "Basic MVO":
            self.sample_data_text.insert(tk.END, "Calculating expected returns and covariance matrix.\n")
            returns = self.data_filtered.mean()
            cov_matrix = self.data_filtered.cov()
            self.sample_data_text.insert(tk.END, "Optimization in progress...\n")

            # Define objective function
            def objective(x):
                portfolio_return = np.dot(x, returns.values)
                portfolio_volatility = np.sqrt(np.dot(x.T, np.dot(cov_matrix.values, x)))
                if portfolio_volatility != 0:
                    if self.optimize_choice.get() == "Maximize Returns":
                        return -portfolio_return
                    elif self.optimize_choice.get() == "Minimize Volatility":
                        return portfolio_volatility
                    else:  # Maximize Sharpe Ratio
                        return -portfolio_return / portfolio_volatility
                else:
                    return np.inf

            # Optimization
            result = minimize(objective, x0, bounds=bounds, constraints=cons)

            if result.success:
                self.sample_data_text.insert(tk.END, "Optimization successful.\n")
                weights = result.x
                # Calculate portfolio metrics
                portfolio_return = np.dot(weights, returns.values)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                if portfolio_volatility != 0:
                    sharpe_ratio = portfolio_return / portfolio_volatility
                else:
                    sharpe_ratio = 0
                # Calculate maximum drawdown for the optimized portfolio
                portfolio_returns = self.data_filtered.dot(weights)
                max_drawdown_value = self.calculate_max_drawdown(portfolio_returns)
            else:
                self.sample_data_text.insert(tk.END, "Optimization failed.\n")
                if max_drawdown is not None:
                    tk.messagebox.showinfo("Optimization Failed", "Cannot achieve the specified maximum drawdown constraint.")
                else:
                    tk.messagebox.showinfo("Optimization Failed", "Optimization failed. Please check your constraints.")
                self.result_text.set("Optimization failed.")
                return

        elif self.model_choice.get() == "Ledoit-Wolf Shrinkage":
            self.sample_data_text.insert(tk.END, "Calculating expected returns.\n")
            returns = self.data_filtered.mean()
            self.sample_data_text.insert(tk.END, "Calculating covariance matrix using Ledoit-Wolf Shrinkage estimator.\n")
            cov_matrix, _ = ledoit_wolf(self.data_filtered)
            cov_matrix = pd.DataFrame(cov_matrix, index=self.fund_names, columns=self.fund_names)
            self.sample_data_text.insert(tk.END, "Optimization in progress...\n")

            # Define objective function
            def objective(x):
                portfolio_return = np.dot(x, returns.values)
                portfolio_volatility = np.sqrt(np.dot(x.T, np.dot(cov_matrix.values, x)))
                if portfolio_volatility != 0:
                    if self.optimize_choice.get() == "Maximize Returns":
                        return -portfolio_return
                    elif self.optimize_choice.get() == "Minimize Volatility":
                        return portfolio_volatility
                    else:  # Maximize Sharpe Ratio
                        return -portfolio_return / portfolio_volatility
                else:
                    return np.inf

            # Optimization
            result = minimize(objective, x0, bounds=bounds, constraints=cons)

            if result.success:
                self.sample_data_text.insert(tk.END, "Optimization successful.\n")
                weights = result.x
                # Calculate portfolio metrics
                portfolio_return = np.dot(weights, returns.values)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                if portfolio_volatility != 0:
                    sharpe_ratio = portfolio_return / portfolio_volatility
                else:
                    sharpe_ratio = 0
                # Calculate maximum drawdown for the optimized portfolio
                portfolio_returns = self.data_filtered.dot(weights)
                max_drawdown_value = self.calculate_max_drawdown(portfolio_returns)
            else:
                self.sample_data_text.insert(tk.END, "Optimization failed.\n")
                if max_drawdown is not None:
                    tk.messagebox.showinfo("Optimization Failed", "Cannot achieve the specified maximum drawdown constraint.")
                else:
                    tk.messagebox.showinfo("Optimization Failed", "Optimization failed. Please check your constraints.")
                self.result_text.set("Optimization failed.")
                return

        elif self.model_choice.get() == "K-Fold Cross-Validation" or self.model_choice.get() == "Combined Ledoit-Wolf and K-Fold":
            num_folds = self.num_folds_var.get()
            self.sample_data_text.insert(tk.END, f"Performing K-Fold Cross-Validation with {num_folds} folds.\n")
            kf = KFold(n_splits=num_folds, shuffle=False)
            weights_list = []
            fold_num = 1
            for train_index, test_index in kf.split(self.data_filtered):
                self.sample_data_text.insert(tk.END, f"Processing fold {fold_num}.\n")
                train_data = self.data_filtered.iloc[train_index]
                # Calculate returns and covariance on training data
                fold_returns = train_data.mean()
                if self.model_choice.get() == "Combined Ledoit-Wolf and K-Fold":
                    self.sample_data_text.insert(tk.END, "Using Ledoit-Wolf Shrinkage estimator for covariance matrix.\n")
                    cov_matrix_fold, _ = ledoit_wolf(train_data)
                    cov_matrix_fold = pd.DataFrame(cov_matrix_fold, index=self.fund_names, columns=self.fund_names)
                else:
                    cov_matrix_fold = train_data.cov()
                # Define objective function
                def fold_objective(x):
                    portfolio_return = np.dot(x, fold_returns.values)
                    portfolio_volatility = np.sqrt(np.dot(x.T, np.dot(cov_matrix_fold.values, x)))
                    if portfolio_volatility != 0:
                        if self.optimize_choice.get() == "Maximize Returns":
                            return -portfolio_return
                        elif self.optimize_choice.get() == "Minimize Volatility":
                            return portfolio_volatility
                        else:  # Maximize Sharpe Ratio
                            return -portfolio_return / portfolio_volatility
                    else:
                        return np.inf
                # Add maximum drawdown constraint
                fold_cons = cons.copy()
                if max_drawdown is not None:
                    def max_drawdown_constraint(x, data=train_data):
                        portfolio_returns = data.dot(x)
                        mdd = self.calculate_max_drawdown(portfolio_returns)
                        return max_drawdown - mdd
                    fold_cons.append({'type': 'ineq', 'fun': max_drawdown_constraint})
                # Optimization
                result_fold = minimize(fold_objective, x0, bounds=bounds, constraints=fold_cons)
                if result_fold.success:
                    weights_list.append(result_fold.x)
                    self.sample_data_text.insert(tk.END, f"Fold {fold_num} optimization successful.\n")
                else:
                    self.sample_data_text.insert(tk.END, f"Fold {fold_num} optimization failed.\n")
                    if max_drawdown is not None:
                        tk.messagebox.showinfo("Optimization Failed", f"Cannot achieve the specified maximum drawdown constraint in fold {fold_num}.")
                    else:
                        tk.messagebox.showinfo("Optimization Failed", f"Optimization failed in fold {fold_num}. Please check your constraints.")
                    self.result_text.set(f"Fold {fold_num} optimization failed.")
                    return
                fold_num += 1
            # Average weights
            weights = np.mean(weights_list, axis=0)
            self.sample_data_text.insert(tk.END, "Averaging weights across folds.\n")

            # Recalculate returns and covariance on the entire dataset
            returns = self.data_filtered.mean()
            cov_matrix = self.data_filtered.cov()
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, returns.values)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            if portfolio_volatility != 0:
                sharpe_ratio = portfolio_return / portfolio_volatility
            else:
                sharpe_ratio = 0
            # Calculate maximum drawdown for the optimized portfolio
            portfolio_returns = self.data_filtered.dot(weights)
            max_drawdown_value = self.calculate_max_drawdown(portfolio_returns)

            self.sample_data_text.insert(tk.END, "Cross-validation optimization completed.\n")
        else:
            self.result_text.set("Invalid optimization model selected.")
            return

        # Ensure that portfolio metrics have been calculated
        if weights is None or portfolio_return is None or portfolio_volatility is None or sharpe_ratio is None:
            self.result_text.set("Optimization did not produce valid results.")
            return

        # Get date range used
        date_range = f"{start_date.strftime('%b-%y')} to {end_date.strftime('%b-%y')}"

        # Store simulation results
        self.simulation_count += 1
        simulation_result = {
            'Simulation': self.simulation_count,
            'Model': self.model_choice.get(),
            'Date': date_range,
            'Weights': dict(zip(self.fund_names, weights)),
            'Constraints': entered_constraints,
            'Asset Allocation Constraints': asset_allocation_constraints,
            'Min Return Constraint': self.min_return_var.get(),
            'Max Drawdown Constraint': self.max_drawdown_var.get(),
            'Sharpe Ratio': sharpe_ratio,
            'Return': portfolio_return,
            'Volatility': portfolio_volatility,
            'Max Drawdown': max_drawdown_value
        }
        self.simulations.append(simulation_result)

        # Display results
        self.display_results(simulation_result)

    def display_results(self, simulation_result):
        sim_num = simulation_result['Simulation']
        col = sim_num - 1  # Zero-based index

        # Create a frame for each simulation
        sim_frame = tk.Frame(self.results_inner_frame, bd=2, relief='groove', padx=5, pady=5)
        sim_frame.grid(row=0, column=col, padx=5, pady=5, sticky='n')

        # Simulation title and date range
        tk.Label(sim_frame, text=f"Simulation {sim_num}", font=('Arial', 12, 'bold')).pack()
        tk.Label(sim_frame, text=f"Model: {simulation_result['Model']}").pack()
        tk.Label(sim_frame, text=f"Date Range: {simulation_result['Date']}").pack()

        # Display Sharpe Ratio, Return, Volatility, and Max Drawdown
        tk.Label(sim_frame, text=f"Sharpe Ratio: {simulation_result['Sharpe Ratio']:.2f}").pack()
        tk.Label(sim_frame, text=f"Return: {simulation_result['Return']*100:.2f}%").pack()
        tk.Label(sim_frame, text=f"Volatility: {simulation_result['Volatility']*100:.2f}%").pack()
        tk.Label(sim_frame, text=f"Max Drawdown: {simulation_result['Max Drawdown']*100:.2f}%").pack()
        tk.Label(sim_frame, text="").pack()  # Spacer

        # Optimized Portfolio Weights
        tk.Label(sim_frame, text="Optimized Portfolio Weights:").pack()
        for fund, weight in simulation_result['Weights'].items():
            tk.Label(sim_frame, text=f"{fund}: {weight*100:.0f}%").pack()

        # Constraints (only those entered)
        if simulation_result['Constraints'] or simulation_result['Asset Allocation Constraints'] or simulation_result['Min Return Constraint'] or simulation_result['Max Drawdown Constraint']:
            tk.Label(sim_frame, text="").pack()  # Spacer
            tk.Label(sim_frame, text="Constraints:").pack()
            for fund, max_weight in simulation_result['Constraints'].items():
                tk.Label(sim_frame, text=f"{fund}: Max {max_weight*100:.0f}%").pack()
            # Display asset allocation constraints
            if simulation_result['Asset Allocation Constraints']:
                for asset_class, allocation in simulation_result['Asset Allocation Constraints'].items():
                    tk.Label(sim_frame, text=f"{asset_class}: {allocation*100:.0f}%").pack()
            # Display additional constraints
            if simulation_result['Min Return Constraint']:
                tk.Label(sim_frame, text=f"Minimum Monthly Return: {simulation_result['Min Return Constraint']}%").pack()
            if simulation_result['Max Drawdown Constraint']:
                tk.Label(sim_frame, text=f"Maximum Drawdown: {simulation_result['Max Drawdown Constraint']}%").pack()

        # Add button to test weights over different period
        test_button = tk.Button(sim_frame, text="Test Weights over Different Period", command=lambda sim=simulation_result: self.test_weights(sim))
        test_button.pack(pady=5)

    def test_weights(self, simulation_result):
        # Create a new window for selecting test date range
        test_window = tk.Toplevel(self.master)
        test_window.title(f"Test Weights for Simulation {simulation_result['Simulation']}")

        # Date selection dropdowns
        tk.Label(test_window, text="Select Test Date Range").pack(pady=5)
        date_frame = tk.Frame(test_window)
        date_frame.pack(pady=5)
        tk.Label(date_frame, text="Start date:").pack(side=tk.LEFT)
        start_date_var = tk.StringVar()
        start_date_dropdown = ttk.Combobox(date_frame, textvariable=start_date_var, width=10, state='readonly')
        start_date_dropdown.pack(side=tk.LEFT, padx=5)

        tk.Label(date_frame, text="End date:").pack(side=tk.LEFT)
        end_date_var = tk.StringVar()
        end_date_dropdown = ttk.Combobox(date_frame, textvariable=end_date_var, width=10, state='readonly')
        end_date_dropdown.pack(side=tk.LEFT, padx=5)

        # Populate date dropdowns with dates from self.data
        dates = self.data.index.strftime('%b-%y').tolist()
        start_date_dropdown['values'] = dates
        end_date_dropdown['values'] = dates

        # Set default selection to earliest and latest dates
        if dates:
            start_date_var.set(dates[0])
            end_date_var.set(dates[-1])

        # Add a button to perform the test
        test_button = tk.Button(test_window, text="Test Weights", command=lambda: self.perform_test(simulation_result, start_date_var.get(), end_date_var.get(), test_window))
        test_button.pack(pady=10)

        # A text area to display results
        test_window.result_text = tk.Text(test_window, height=10, width=50)
        test_window.result_text.pack(pady=5)

    def perform_test(self, simulation_result, start_date_str, end_date_str, test_window):
        try:
            # Parse dates
            start_date = datetime.strptime(start_date_str, '%b-%y')
            end_date = datetime.strptime(end_date_str, '%b-%y')

            if start_date > end_date:
                tk.messagebox.showinfo("Invalid Date Range", "Start date must be before End date.")
                return

            # Filter data
            data_filtered = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]

            if data_filtered.empty:
                tk.messagebox.showinfo("No Data", "No data available for the selected date range.")
                return

            # Get the weights from the simulation
            weights = simulation_result['Weights']
            # Convert weights dict to numpy array in the correct order
            weights_array = np.array([weights[fund] for fund in self.fund_names])

            # Calculate portfolio returns
            portfolio_returns = data_filtered.dot(weights_array)

            # Calculate performance metrics
            portfolio_return = portfolio_returns.mean()
            portfolio_volatility = portfolio_returns.std()
            if portfolio_volatility != 0:
                sharpe_ratio = portfolio_return / portfolio_volatility
            else:
                sharpe_ratio = 0
            max_drawdown_value = self.calculate_max_drawdown(portfolio_returns)

            # Display results
            test_window.result_text.delete('1.0', tk.END)
            test_window.result_text.insert(tk.END, f"Performance of Simulation {simulation_result['Simulation']} Weights\n")
            test_window.result_text.insert(tk.END, f"Test Date Range: {start_date_str} to {end_date_str}\n")
            test_window.result_text.insert(tk.END, f"Return: {portfolio_return*100:.2f}%\n")
            test_window.result_text.insert(tk.END, f"Volatility: {portfolio_volatility*100:.2f}%\n")
            test_window.result_text.insert(tk.END, f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
            test_window.result_text.insert(tk.END, f"Max Drawdown: {max_drawdown_value*100:.2f}%\n")

        except Exception as e:
            tk.messagebox.showinfo("Error", f"An error occurred: {e}")
            return

    def calculate_max_drawdown(self, returns_series):
        cumulative = (1 + returns_series).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        return abs(max_drawdown)

    def export_results(self):
        if not self.simulations:
            tk.messagebox.showinfo("No Simulations", "There are no simulations to export.")
            return

        # Ask user where to save the Excel file
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")],
                                                 title="Save Simulations")
        if file_path:
            # Prepare data for export
            metrics = ['Model', 'Date Range', 'Sharpe Ratio', 'Return (%)', 'Volatility (%)', 'Max Drawdown (%)'] + self.fund_names
            data = []
            for sim in self.simulations:
                sim_data = {
                    'Model': sim['Model'],
                    'Date Range': sim['Date'],
                    'Sharpe Ratio': sim['Sharpe Ratio'],
                    'Return (%)': sim['Return'] * 100,
                    'Volatility (%)': sim['Volatility'] * 100,
                    'Max Drawdown (%)': sim['Max Drawdown'] * 100,
                }
                for fund in self.fund_names:
                    sim_data[fund] = sim['Weights'].get(fund, 0) * 100  # Convert to percentage
                data.append(sim_data)

            df_export = pd.DataFrame(data, columns=metrics)
            df_export.index += 1  # Start simulation numbering from 1
            df_export.index.name = 'Simulation'

            # Write to Excel without any formatting
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, sheet_name='Simulations')

                # Constraints data
                nrows = df_export.shape[0] + 2  # Leave a row empty after simulations
                constraints_data = []
                for sim in self.simulations:
                    sim_constraints = {}
                    for fund in self.fund_names:
                        max_weight = sim['Constraints'].get(fund, 1) * 100  # Convert to percentage
                        sim_constraints[fund] = max_weight
                    # Add asset allocation constraints
                    for asset_class in self.asset_classes_list:
                        allocation = sim['Asset Allocation Constraints'].get(asset_class, "")
                        sim_constraints[asset_class + ' Allocation (%)'] = allocation * 100 if allocation != "" else ""
                    # Add additional constraints
                    sim_constraints['Minimum Monthly Return (%)'] = sim['Min Return Constraint']
                    sim_constraints['Maximum Drawdown (%)'] = sim['Max Drawdown Constraint']
                    constraints_data.append(sim_constraints)

                constraints_columns = ['Minimum Monthly Return (%)', 'Maximum Drawdown (%)'] + \
                                      [asset_class + ' Allocation (%)' for asset_class in self.asset_classes_list] + self.fund_names
                df_constraints = pd.DataFrame(constraints_data, columns=constraints_columns)
                df_constraints.index += 1  # Simulation numbering
                df_constraints.index.name = 'Simulation'

                df_constraints.to_excel(writer, sheet_name='Simulations', startrow=nrows)

            tk.messagebox.showinfo("Export Successful", f"Simulations exported to {file_path}")

    def generate_efficient_frontier(self):
        tk.messagebox.showinfo("Feature Not Implemented", "Efficient Frontier generation is not updated to handle asset allocation constraints in this version.")

    def clear_all_outputs(self):
        # Clear all constraints
        self.clear_constraints()

        # Reset simulations and simulation count
        self.simulations.clear()
        self.simulation_count = 0

        # Clear the results display
        for widget in self.results_inner_frame.winfo_children():
            widget.destroy()

        # Clear the efficient frontier chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Reset fund stats labels
        for fund in self.fund_names:
            self.fund_stats[fund]['ReturnLabel'].config(text="0.00%")
            self.fund_stats[fund]['VolatilityLabel'].config(text="0.00%")

        # Reset date selections
        if self.start_date_dropdown['values']:
            self.start_date_var.set(self.start_date_dropdown['values'][0])
        if self.end_date_dropdown['values']:
            self.end_date_var.set(self.end_date_dropdown['values'][-1])

        tk.messagebox.showinfo("Cleared", "All constraints and outputs have been cleared.")

    # Mouse wheel scrolling for results canvas
    def _on_mousewheel_results(self, event):
        self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_shift_mousewheel_results(self, event):
        self.results_canvas.xview_scroll(int(-1*(event.delta/120)), "units")

    def open_backtest_window(self):
        if not self.simulations:
            tk.messagebox.showinfo("No Simulations", "Please generate simulations first.")
            return

        # Create a new window for backtesting
        backtest_window = tk.Toplevel(self.master)
        backtest_window.title("Backtest Simulations")

        # Simulation selection
        tk.Label(backtest_window, text="Select Simulation to Backtest:").pack(pady=5)
        sim_names = [f"Simulation {sim['Simulation']}" for sim in self.simulations]
        self.backtest_sim_var = tk.StringVar()
        sim_dropdown = ttk.Combobox(backtest_window, textvariable=self.backtest_sim_var, values=sim_names, state='readonly')
        sim_dropdown.pack(pady=5)

        # Start date selection
        tk.Label(backtest_window, text="Select Simulation Start Date:").pack(pady=5)
        dates = self.data.index.strftime('%b-%Y').tolist()
        self.backtest_start_date_var = tk.StringVar()
        start_date_dropdown = ttk.Combobox(backtest_window, textvariable=self.backtest_start_date_var, values=dates, width=10, state='readonly')
        start_date_dropdown.pack(pady=5)
        if dates:
            self.backtest_start_date_var.set(dates[0])

        # Benchmark selection
        tk.Label(backtest_window, text="Select Benchmarks (Max 2):").pack(pady=5)
        benchmark_funds = [fund for fund in self.fund_names if self.asset_classes.get(fund) == "Benchmark"]
        self.backtest_benchmark_vars = []
        for i in range(2):
            var = tk.StringVar()
            benchmark_dropdown = ttk.Combobox(backtest_window, textvariable=var, values=benchmark_funds, state='readonly')
            benchmark_dropdown.pack(pady=5)
            self.backtest_benchmark_vars.append(var)

        # Run backtest button
        run_button = tk.Button(backtest_window, text="Run Backtest", command=lambda: self.run_backtest(backtest_window))
        run_button.pack(pady=10)

    def run_backtest(self, backtest_window):       
        sim_name = self.backtest_sim_var.get()
        if not sim_name:
            tk.messagebox.showinfo("No Simulation Selected", "Please select a simulation to backtest.")
            return
        sim_index = int(sim_name.split()[-1]) - 1
        if 0 <= sim_index < len(self.simulations):
            simulation_result = self.simulations[sim_index]
        else:
            tk.messagebox.showinfo("Invalid Simulation", "Selected simulation is invalid.")
            return

        # Get simulation start date
        start_date_str = self.backtest_start_date_var.get()
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%b-%Y')
            except Exception as e:
                tk.messagebox.showinfo("Invalid Date", f"Invalid start date: {e}")
                return
        else:
            tk.messagebox.showinfo("No Start Date", "Please select a simulation start date.")
            return

        # Get selected benchmarks
        benchmarks = []
        for var in self.backtest_benchmark_vars:
            benchmark = var.get()
            if benchmark:
                benchmarks.append(benchmark)
        if len(benchmarks) > 2:
            tk.messagebox.showinfo("Too Many Benchmarks", "Please select up to 2 benchmarks.")
            return

        # Filter data from start date
        data_filtered = self.data[self.data.index >= start_date]
        if data_filtered.empty:
            tk.messagebox.showinfo("No Data", "No data available from the selected start date.")
            return

        # Add a canvas for scrolling
        canvas = tk.Canvas(backtest_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar and attach it to the canvas
        scrollbar = tk.Scrollbar(backtest_window, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")

        # Configure canvas to update scroll region
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Add a frame inside the canvas to hold all widgets
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # Display the backtest results within this scrollable frame
        self.perform_backtest(simulation_result, data_filtered, benchmarks, frame)
        

    def perform_backtest(self, simulation_result, data_filtered, benchmarks, backtest_window):
        # Initialize portfolio value
        initial_investment = 1000.0
        portfolio_values = pd.Series(index=data_filtered.index)
        portfolio_values.iloc[0] = initial_investment

        # Get weights from the simulation
        weights = simulation_result['Weights']
        target_weights = {fund: weights.get(fund, 0.0) for fund in self.fund_names}
        total_weight = sum(target_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            tk.messagebox.showinfo("Invalid Weights", "Total weights do not sum up to 100%.")
            return

        # Initialize weights
        weights_df = pd.DataFrame(index=data_filtered.index, columns=self.fund_names)
        weights_df.iloc[0] = target_weights

        # Calculate portfolio returns and rebalance annually
        for i in range(1, len(data_filtered)):
            date = data_filtered.index[i]
            prev_date = data_filtered.index[i-1]
            returns = data_filtered.iloc[i]
            prev_weights = weights_df.iloc[i-1]
            portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + np.dot(prev_weights, returns))

            # Check if end of year to rebalance
            if date.year != prev_date.year:
                weights_df.iloc[i] = target_weights  # Rebalance
            else:
                # Keep previous weights adjusted for returns
                new_weights = prev_weights * (1 + returns)
                new_weights = new_weights / new_weights.sum()
                weights_df.iloc[i] = new_weights

        # Calculate cumulative returns for benchmarks
        benchmark_values = {}
        for benchmark in benchmarks:
            benchmark_returns = data_filtered[benchmark]
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() * initial_investment
            benchmark_values[benchmark] = benchmark_cum_returns

        # Plot the results
        self.plot_backtest_results(portfolio_values, benchmark_values, backtest_window)

        # Calculate performance metrics
        self.calculate_backtest_metrics(portfolio_values, benchmark_values, data_filtered, backtest_window)

    def plot_backtest_results(self, portfolio_values, benchmark_values, backtest_window):
        fig, ax = plt.subplots(figsize=(8, 6))
        portfolio_values.plot(ax=ax, label='Portfolio')
        for benchmark, values in benchmark_values.items():
            values.plot(ax=ax, label=benchmark)
        ax.set_title('Portfolio Value Over Time')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()

        # Display chart in backtest window
        chart_frame = tk.Frame(backtest_window)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def calculate_backtest_metrics(self, portfolio_values, benchmark_values, data_filtered, backtest_window):
        # Calculate portfolio metrics
        portfolio_returns = portfolio_values.pct_change().dropna()
        annualized_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (12 / len(portfolio_returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(12)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_returns)

        # Calendar Year Performance for Portfolio
        portfolio_yearly_returns = portfolio_returns.resample('Y').apply(lambda r: (1 + r).prod() - 1)

        # Benchmark metrics and calendar year returns
        benchmark_annualized_metrics = {}
        benchmark_yearly_returns = {}
        for benchmark, values in benchmark_values.items():
            returns = values.pct_change().dropna()
            benchmark_annualized_metrics[benchmark] = {
                "Annualized Return": (values.iloc[-1] / values.iloc[0]) ** (12 / len(returns)) - 1,
                "Annualized Volatility": returns.std() * np.sqrt(12),
            }
            benchmark_yearly_returns[benchmark] = returns.resample('Y').apply(lambda r: (1 + r).prod() - 1)

        # Display metrics and prepare table structure
        metrics_text = f"Portfolio:\n" \
                       f"Annualized Return: {annualized_return * 100:.2f}\n" \
                       f"Annualized Volatility: {annualized_volatility * 100:.2f}\n" \
                       f"Sharpe Ratio: {sharpe_ratio:.2f}\n" \
                       f"Maximum Drawdown: {max_drawdown * 100:.2f}\n\n"

        # Create table for calendar year returns
        metrics_text += "Calendar Year Returns:\n"
        years = sorted(set(portfolio_yearly_returns.index.year).union(
            *[returns.index.year for returns in benchmark_yearly_returns.values()]))

        # Table Header
        metrics_text += f"{'Year':<10}| {'Portfolio':<12}"
        for benchmark in benchmark_yearly_returns.keys():
            metrics_text += f"| {benchmark:<12}"
        metrics_text += "\n" + "-" * (13 + 13 * (1 + len(benchmark_yearly_returns))) + "\n"

        # Fill table rows with returns for each year, handling NaNs
        for year in years:
            year_str = f"{year:<10}"
            portfolio_return = portfolio_yearly_returns.loc[portfolio_yearly_returns.index.year == year]
            row = f"{year_str}| {f'{float(portfolio_return.iloc[0]) * 100:.2f}' if not portfolio_return.empty else '--':<12}"

            for benchmark, returns in benchmark_yearly_returns.items():
                benchmark_return = returns.loc[returns.index.year == year]
                row += f"| {f'{float(benchmark_return.iloc[0]) * 100:.2f}' if not benchmark_return.empty else '--':<12}"

            metrics_text += row + "\n"

        # Annualized Volatility and Returns Table
        metrics_text += "\nAnnualized Metrics:\n"
        metrics_text += f"{'Metric':<20}| {'Portfolio':<12}"
        for benchmark in benchmark_annualized_metrics.keys():
            metrics_text += f"| {benchmark:<12}"
        metrics_text += "\n" + "-" * (25 + 13 * (1 + len(benchmark_annualized_metrics))) + "\n"

        # Add rows for Annualized Return and Volatility
        annualized_return_row = f"{'Annualized Return':<20}| {annualized_return * 100:.2f}"
        annualized_volatility_row = f"{'Annualized Volatility':<20}| {annualized_volatility * 100:.2f}"
        for benchmark, metrics in benchmark_annualized_metrics.items():
            annualized_return_row += f"| {metrics['Annualized Return'] * 100:.2f}"
            annualized_volatility_row += f"| {metrics['Annualized Volatility'] * 100:.2f}"

        metrics_text += annualized_return_row + "\n" + annualized_volatility_row + "\n"

        # Display metrics in the backtest window
        metrics_frame = tk.Frame(backtest_window)
        metrics_frame.pack(pady=5)
        tk.Label(metrics_frame, text="Performance Metrics", font=('Arial', 12, 'bold')).pack()
        metrics_label = tk.Label(metrics_frame, text=metrics_text, justify=tk.LEFT, font=('Courier', 10))
        metrics_label.pack()
        

# Run the app
if __name__ == '__main__':
    root = tk.Tk()
    app = MVOApp(root)
    root.mainloop()


# In[ ]:




