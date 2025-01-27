import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# -> Uncomment to suppress a warning that occurs when running the program.
# -> The message warns about making sure to check your CSV file for empty cells
# -> regardless of whether or not there are any empty cells in the file.
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# CONSTANTS
DEFAULT_CONCENTRATION = 0.5
DEFAULT_FLOW_RATE = 120
FLOW_RATE_RATIO = 38 / 120
NOISE_VARIATION = 0.05
LEVELS = 5
SPLITS = 3

# UTILITY FUNCTIONS

# Runs simulation based on given parameters
def run_simulation(model, initial_conc, initial_rate, levels, data_columns):
    nodes = [{
        "device_id": 1, "level": 1, "parent_id": None,
        "starting_concentration": initial_conc, "flow_rate": initial_rate
    }]

    simulation_results = pd.DataFrame(columns=[
        "Device", "Level", "Parent_Device", "Starting_Concentration",
        "Predicted_Left_Out_perc", "Predicted_Right_Out_perc", "Flow_Rate_ml_min"
    ])

    device_id = 1

    while nodes:
        next_nodes = []
        for node in nodes:
            device_data = pd.DataFrame({
                "Starting_Concentration": [node["starting_concentration"]],
                "Flow_Rate_ml_min": [node["flow_rate"]]
            }, columns=data_columns)

            predicted_values = model.predict(device_data)[0]
            predicted_left_out, predicted_right_out = predicted_values

            simulation_results = pd.concat([
                simulation_results,
                pd.DataFrame({
                    "Device": [node["device_id"]],
                    "Level": [node["level"]],
                    "Parent_Device": [node["parent_id"]],
                    "Starting_Concentration": [node["starting_concentration"]],
                    "Predicted_Left_Out_perc": [predicted_left_out],
                    "Predicted_Right_Out_perc": [predicted_right_out],
                    "Flow_Rate_ml_min": [node["flow_rate"]]
                })
            ], ignore_index=True)

            if node["level"] < levels:
                for i, child_concentration in enumerate([predicted_left_out, predicted_right_out]):
                    device_id += 1
                    child_flow_rate = FLOW_RATE_RATIO * node["flow_rate"]
                    next_nodes.append({
                        "device_id": device_id,
                        "level": node["level"] + 1,
                        "parent_id": node["device_id"],
                        "starting_concentration": child_concentration,
                        "flow_rate": child_flow_rate
                    })
        nodes = next_nodes

    return simulation_results

# Builds tree data based on simulations
def build_tree_data(simulation_results, device_id=1, level=1, max_level=None):
    device_row = simulation_results[simulation_results["Device"] == device_id]
    if device_row.empty:
        return None

    starting_concentration = device_row.iloc[0]["Starting_Concentration"]

    if max_level is not None and level > max_level:
        return {"value": f"{starting_concentration:.2f}"}

    children = simulation_results[simulation_results["Parent_Device"] == device_id]
    left_child = children.iloc[0]["Device"] if len(children) > 0 else None
    right_child = children.iloc[1]["Device"] if len(children) > 1 else None

    return {
        "value": f"{starting_concentration:.2f}",
        "left": build_tree_data(simulation_results, left_child, level + 1, max_level) if left_child else None,
        "right": build_tree_data(simulation_results, right_child, level + 1, max_level) if right_child else None,
    }

# Prints binary tree in correct format
def print_tree(tree, level=0, width=80):
    if tree is None:
        return []

    left_lines = print_tree(tree.get("left"), level + 1, width // 2)
    right_lines = print_tree(tree.get("right"), level + 1, width // 2)
    current_line = f"{tree['value']}".center(width)

    combined_lines = []
    max_height = max(len(left_lines), len(right_lines))
    for i in range(max_height):
        left_line = left_lines[i] if i < len(left_lines) else " " * (width // 2)
        right_line = right_lines[i] if i < len(right_lines) else " " * (width // 2)
        combined_lines.append(left_line + right_line)

    return [current_line] + combined_lines

# Load .CSV
data = pd.read_csv('Full_Data_FINAL.csv')

# Prepare Data for Model
X = data.drop(columns=['Left_Out_perc', 'Right_Out_perc'])
y = data[['Left_Out_perc', 'Right_Out_perc']]

rf_model = RandomForestRegressor(random_state=0)
for i in range(SPLITS):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    # -> uncomment to see MSE
    #print(f"Random Forest MSE (Split {i+1}): {rf_mse}")

# GUI of application
class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soret Effect Device Network Simulation")

        # Input
        self.input_frame = ttk.Frame(self.root, padding="10")
        self.input_frame.grid(row=0, column=0, sticky=tk.W)

        ttk.Label(self.input_frame, text="Initial Starting Concentration:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_conc_entry = ttk.Entry(self.input_frame, width=10)
        self.start_conc_entry.grid(row=0, column=1, padx=5, pady=5)
        self.start_conc_entry.insert(0, str(DEFAULT_CONCENTRATION))
        ttk.Label(self.input_frame, text="Hydrogen proportion").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

        ttk.Label(self.input_frame, text="Initial Flow Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.flow_rate_entry = ttk.Entry(self.input_frame, width=10)
        self.flow_rate_entry.grid(row=1, column=1, padx=5, pady=5)
        self.flow_rate_entry.insert(0, str(DEFAULT_FLOW_RATE))
        ttk.Label(self.input_frame, text="ml/min").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

        ttk.Label(self.input_frame, text="Depth of Network:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.levels_var = tk.StringVar(value=str(LEVELS))
        self.levels_slider = ttk.Scale(self.input_frame, from_=1, to=LEVELS, orient=tk.HORIZONTAL, length=150, command=self.update_slider_label)
        self.levels_slider.grid(row=2, column=1, padx=5, pady=5)
        self.levels_slider.set(LEVELS)
        ttk.Label(self.input_frame, textvariable=self.levels_var).grid(row=2, column=2, padx=5, pady=5)

        self.run_button = ttk.Button(self.input_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Output
        self.output_frame = ttk.Frame(self.root, padding="10")
        self.output_frame.grid(row=1, column=0, sticky=tk.W)

        ttk.Label(self.output_frame, text="Device Network Structure:").grid(row=0, column=0, sticky=tk.W)
        self.tree_display = scrolledtext.Text(self.output_frame, width=81, height=8)
        self.tree_display.grid(row=1, column=0, padx=5, pady=5)

    # Function for dynamic slider value
    def update_slider_label(self, value):
        self.levels_var.set(f"{int(float(value))}")

    def run_simulation(self):
        try:
            initial_conc = float(self.start_conc_entry.get())
            initial_rate = float(self.flow_rate_entry.get())
            levels = int(self.levels_slider.get())

            if initial_conc < 0.0001 or initial_conc > 0.9999:
                raise ValueError("Starting Concentration must be between 0.0001 and 0.9999.")
            if initial_rate < 30 or initial_rate > 180:
                raise ValueError("Flow Rate must be between 30 and 200.")

            simulation_results = run_simulation(rf_model, initial_conc, initial_rate, levels, X.columns)
            tree_data = build_tree_data(simulation_results, device_id=1, max_level=levels)
            tree_lines = print_tree(tree_data, width=80)

            self.tree_display.delete('1.0', tk.END)
            self.tree_display.insert(tk.END, "\n".join(tree_lines))

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()
