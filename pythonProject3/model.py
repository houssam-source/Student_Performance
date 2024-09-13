import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Global variables to hold data and model
data_frame = None
lin_reg_model = None


def import_csv():
    """Function to import CSV file and load it into a pandas DataFrame."""
    global data_frame
    try:
        # Open file dialog to select CSV file
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        # Read the CSV file using pandas
        data_frame = pd.read_csv(file_path)

        # Display basic information about the dataset
        messagebox.showinfo("Success", f"CSV file imported successfully!\n\nColumns:\n{', '.join(data_frame.columns)}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def train_model():
    """Function to train the Linear Regression model and display results."""
    global lin_reg_model, data_frame
    if data_frame is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return
    try:
        # Update these columns to match the actual dataset
        feature_columns = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities',
                           'ParentalSupport']
        target_column = 'FinalGrade'

        # Verify if all the required columns exist in the dataset
        missing_cols = [col for col in feature_columns + [target_column] if col not in data_frame.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Missing columns in dataset: {', '.join(missing_cols)}")
            return

        # Remove rows with missing target values (FinalGrade) and handle missing features
        data_frame_clean = data_frame.dropna(subset=[target_column])
        data_frame_clean = data_frame_clean.dropna(subset=feature_columns)

        # Prepare features (X) and target (y)
        X = data_frame_clean[feature_columns]
        y = data_frame_clean[target_column]  # Target (what you want to predict)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Initialize and train the linear regression model
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, y_train)

        # Predict using the test set
        predictions = lin_reg_model.predict(X_test)

        # Predict final grade for each student in the entire dataset
        full_predictions = lin_reg_model.predict(X)

        # Add the predictions to the original data frame
        data_frame_clean['PredictedFinalGrade'] = full_predictions

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)

        # Display predictions, MSE, and model coefficients
        predictions_text.set(
            f"Predicted Final Grades (for test set): {predictions[:5]}...")  # Display first 5 for brevity
        mse_text.set(f"Mean Squared Error: {mse:.4f}")
        coef_text.set(f"Model Coefficients (Linear Regression): {lin_reg_model.coef_}")
        intercept_text.set(f"Model Intercept (Linear Regression): {lin_reg_model.intercept_}")

        # Save the predictions to a CSV file
        data_frame_clean.to_csv("student_performance_with_predictions.csv", index=False)
        messagebox.showinfo("Success", "Model trained and predictions saved to CSV.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox


def plot_scatter_matrix():
    """Function to plot a scatter matrix for the features in the dataset."""
    if data_frame is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    try:
        # Define the numeric columns for plotting
        feature_columns = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities',
                           'ParentalSupport']

        # Check if all required columns exist in the DataFrame
        missing_cols = [col for col in feature_columns if col not in data_frame.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Missing columns for plotting: {', '.join(missing_cols)}")
            return

        # Ensure all plotting columns are numeric
        plot_data = data_frame[feature_columns].dropna()  # Drop rows with NaN values in these columns

        # Print data types for debugging
        print("Data Types of Columns for Plotting:")
        print(plot_data.dtypes)

        # Check if all columns are numeric
        if not all(pd.api.types.is_numeric_dtype(plot_data[col]) for col in feature_columns):
            messagebox.showerror("Error", "All columns for plotting must be numeric.")
            return

        # Debug: Print a preview of the data to ensure correctness
        print("Data Preview for Plotting:")
        print(plot_data.head())

        # Plot scatter matrix using seaborn for better color handling
        num_features = len(feature_columns)
        fig, axes = plt.subplots(nrows=num_features, ncols=num_features, figsize=(12, 12))

        # Flatten axes array for easier iteration
        axes = axes.flatten()

        # Loop through each pair of features to create scatter plots
        for i in range(num_features):
            for j in range(num_features):
                ax = axes[i * num_features + j]

                if i == j:
                    # Diagonal: Plot KDE
                    sns.kdeplot(data=plot_data, x=feature_columns[i], ax=ax, fill=True, color='blue')
                else:
                    # Off-diagonal: Scatter plots
                    sns.scatterplot(data=plot_data, x=feature_columns[j], y=feature_columns[i], ax=ax, color='blue',
                                    s=30, alpha=0.5)

                # Set labels and titles
                if i == num_features - 1:
                    ax.set_xlabel(feature_columns[j])
                if j == 0:
                    ax.set_ylabel(feature_columns[i])

                # Optional: Add a legend (if applicable)
                if i == 0 and j == 0:
                    ax.legend(['Data Points'], loc='upper right')

        plt.suptitle('Scatter Matrix of Features', fontsize=16)

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust top to fit the suptitle

        # Embed the plot in Tkinter window
        for widget in plot_frame.winfo_children():
            widget.destroy()  # Clear any existing widgets in plot_frame

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while plotting data: {e}")


# Create the main window
root = tk.Tk()
root.title("Advanced Student Performance Prediction")

# Create a tabbed interface
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Tab for data import and model training
frame_train = ttk.Frame(notebook)
notebook.add(frame_train, text="Data Import & Train Model")

# Tab for plotting data
plot_frame = ttk.Frame(notebook)
notebook.add(plot_frame, text="Data Visualization")

# Frame Layout for Import and Training
frame_import = ttk.LabelFrame(frame_train, text="Import Data")
frame_import.pack(fill=tk.X, padx=10, pady=10)

frame_train_model = ttk.LabelFrame(frame_train, text="Train Model & Results")
frame_train_model.pack(fill=tk.X, padx=10, pady=10)

# Buttons for data import and training
btn_import = ttk.Button(frame_import, text="Import CSV", command=import_csv)
btn_import.pack(pady=10)

btn_train = ttk.Button(frame_train_model, text="Train Model", command=train_model)
btn_train.pack(pady=10)

# Output Labels
normal_eq_text = tk.StringVar()
label_normal_eq = tk.Label(frame_train_model, textvariable=normal_eq_text, wraplength=400)
label_normal_eq.pack(pady=5)

predictions_text = tk.StringVar()
label_predictions = tk.Label(frame_train_model, textvariable=predictions_text, wraplength=400)
label_predictions.pack(pady=5)

mse_text = tk.StringVar()
label_mse = tk.Label(frame_train_model, textvariable=mse_text, wraplength=400)
label_mse.pack(pady=5)

coef_text = tk.StringVar()
label_coef = tk.Label(frame_train_model, textvariable=coef_text, wraplength=400)
label_coef.pack(pady=5)

intercept_text = tk.StringVar()
label_intercept = tk.Label(frame_train_model, textvariable=intercept_text, wraplength=400)
label_intercept.pack(pady=5)

# Button to plot data
btn_plot = ttk.Button(plot_frame, text="Plot Scatter Matrix", command=plot_scatter_matrix)
btn_plot.pack(pady=10)

# Run the GUI loop
root.mainloop()
