import pandas as pd

# Read the input CSV file
df = pd.read_csv("Old_Data.csv", delimiter=',')

# Create a list to hold the new data
new_data = []

# Iterate through each row in the DataFrame, performing forward and reverse calculations
# This process produces some duplicates, which were removed afterwards
# Left_Out_perc - 'High' hydrogen end prediction of device
# Right_Out_perc - 'Low' hydrogen end prediction of device
for index, row in df.iterrows():

    # Base value to extrapolate
    starting_val = row["Left_Out_perc"]
    # Determining the relative change in the dataset between the starting and ending concentrations
    increment = row["Left_Out_perc"] - row["Starting_Concentration"]
    
    # Append the original row to the new data
    new_data.append({
        "Flow_Rate_ml_min": row["Flow_Rate_ml_min"],
        "Starting_Concentration": row["Starting_Concentration"],
        "Left_Out_perc": row["Left_Out_perc"]
    })
    
    # Forward calculation
    forward_starting_val = starting_val
    while (forward_starting_val + increment) < 1:
        new_row = {
            "Flow_Rate_ml_min": row["Flow_Rate_ml_min"],
            "Starting_Concentration": forward_starting_val,
            "Left_Out_perc": forward_starting_val + increment
        }
        new_data.append(new_row)
        forward_starting_val += increment

    # Reverse calculation
    reverse_starting_val = starting_val
    while (reverse_starting_val - increment) > 0:
        new_row = {
            "Flow_Rate_ml_min": row["Flow_Rate_ml_min"],
            "Starting_Concentration": reverse_starting_val - increment,
            "Left_Out_perc": reverse_starting_val
        }
        new_data.append(new_row)
        reverse_starting_val -= increment

# Defining the function that calculates Right_Out_perc based on Flow_Rate_ml_min (x) and Left_Out_perc (y)
# -> Since flow rate was what determined the separation efficiency, and since the separation was asymetrical,
# -> the author created another best fit line based on existing data, and then applied it to the Left_Out_Perc
# -> collumn to determine the 'Right_Out_Perc' value
def calculate_right_out_perc(x, y):
    return y - 0.1537876 + 0.00005798503 * x - 0.000001621239 * x ** 2

# Apply the function to fill the Right_Out_perc column
df['Right_Out_perc'] = df.apply(lambda row: calculate_right_out_perc(row['Flow_Rate_ml_min'], row['Left_Out_perc']), axis=1)

# Convert the new data into a DataFrame
new_df = pd.DataFrame(new_data)

# Save the modified dataset to a new CSV file
new_df.to_csv("New_File.csv", index=False)

# Inform user of completion of task
print("Modified dataset saved as 'New_File.csv'.")


''' Pseudocode

Create dataset from csv;
For each row X:
	starting_val = Row X Collumn C;
    # relative change calculation
	increment = (Row X Collumn C) - (Row X Collumn B);
    # before the resulting concentration reaches 1 
	While ((starting_val + increment) < 1)
		Append the dataset with a row as follows:
			Collumn A: Row X Collumn A;
			Collumn B: starting_val;
			Collumn C: starting_val + increment;
		starting_val += increment;

Apply best fit line for right exit to collumn
        
Save dataset to new csv titled "Fake_Data_3";

'''
