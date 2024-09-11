import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Global variables to hold data and model
data_frame = None
lin_reg_model = None


def import_csv():
    global data_frame
    try:
        # Open file dialog to select CSV file
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        # Read the CSV file using pandas
        data_frame = pd.read_csv(file_path)
        messagebox.showinfo("Success", "CSV file imported successfully!")
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")





def train_model():
    global lin_reg_model, data_frame
    if data_frame is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    try:
        # Assuming 'FinalGrade' is the target and other columns are features
        X = data_frame[['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities']]
        y = data_frame['FinalGrade']  # Target (what you want to predict)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Initialize and train the linear regression model
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, y_train)

        # Predict using the test set
        predictions = lin_reg_model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)

        # Display predictions and MSE
        predictions_text.set(f"Predicted Final Grades: {predictions}")
        mse_text.set(f"Mean Squared Error: {mse}")

#plot
        data_frame.plot(x='AttendanceRate', y ='FinalGrade')
        plt.show()
        # Optional: Print the model's coefficients
        coef_text.set(f"Model Coefficients: {lin_reg_model.coef_}")
        intercept_text.set(f"Model Intercept: {lin_reg_model.intercept_}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")


# Create the main window
root = tk.Tk()
root.title("Student Performance Prediction")

# Button to import CSV
btn_import = tk.Button(root, text="Import CSV", command=import_csv)
btn_import.pack(pady=10)

# Button to train model
btn_train = tk.Button(root, text="Train Model", command=train_model)
btn_train.pack(pady=10)

# Labels to display output
predictions_text = tk.StringVar()
label_predictions = tk.Label(root, textvariable=predictions_text, wraplength=400)
label_predictions.pack(pady=10)

mse_text = tk.StringVar()
label_mse = tk.Label(root, textvariable=mse_text, wraplength=400)
label_mse.pack(pady=10)

coef_text = tk.StringVar()
label_coef = tk.Label(root, textvariable=coef_text, wraplength=400)
label_coef.pack(pady=10)

intercept_text = tk.StringVar()
label_intercept = tk.Label(root, textvariable=intercept_text, wraplength=400)
label_intercept.pack(pady=10)

# Run the GUI loop
root.mainloop()
