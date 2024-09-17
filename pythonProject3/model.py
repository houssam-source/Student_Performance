import pandas as pd
from tkinter import filedialog, messagebox
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Global variable to hold the loaded dataset
data_frame = None

# CSV Import function
def import_csv():
    """Function to import CSV file and load it into a pandas DataFrame."""
    global data_frame
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        data_frame = pd.read_csv(file_path)
        messagebox.showinfo("Success", f"CSV file imported successfully!\n\nColumns:\n{', '.join(data_frame.columns)}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Model training function using Linear Regression
def train_improved_model():
    """Train the model using Linear Regression and optimize the loss function (MSE)."""
    global data_frame
    try:
        feature_columns = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities',
                           'ParentalSupport', 'Gender']
        target_column = 'FinalGrade'

        # Impute missing target values
        imputer = SimpleImputer(strategy='mean')
        data_frame_clean = data_frame.copy()
        data_frame_clean[target_column] = imputer.fit_transform(data_frame_clean[[target_column]])

        # Prepare features (X) and target (y)
        X = data_frame_clean[feature_columns]
        y = data_frame_clean[target_column]

        # One-hot encode the 'Gender' feature
        column_transformer = ColumnTransformer(transformers=[
            ('num', StandardScaler(), feature_columns[:-1]),  # Scale numeric features
            ('cat', OneHotEncoder(), ['Gender'])  # One-hot encode gender
        ])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Create a Linear Regression pipeline
        pipe = Pipeline(steps=[
            ('preprocessor', column_transformer),
            ('linear_reg', LinearRegression())  # Linear Regression
        ])

        # Fit the model on training data
        pipe.fit(X_train, y_train)

        # Predict on test data
        predictions = pipe.predict(X_test)

        # Predict on the full dataset (including missing final grades)
        full_predictions = pipe.predict(X)

        # Add the predictions to the original DataFrame
        data_frame_clean['PredictedFinalGrade'] = full_predictions

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)

        # Save predictions to CSV
        data_frame_clean.to_csv("student_performance_with_improved_predictions.csv", index=False)
        messagebox.showinfo("Success", f"Model trained with MSE: {mse:.4f}")

        # Display results in the GUI
        mse_text.set(f"Mean Squared Error (MSE): {mse:.4f}")
        theta_text.set(f"Model Coefficients (Theta): {pipe.named_steps['linear_reg'].coef_}")
        prediction_model_text.set("Model: Linear Regression")

        # Show the predicted final grades
        results_text.set(data_frame_clean[['PredictedFinalGrade']].head().to_string())  # Show top 5 predictions

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")

# Improved Data Visualization Function
def plot_improved_scatter():
    """Simplified and user-friendly scatter matrix for key features."""
    global data_frame
    if data_frame is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    try:
        important_features = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade']
        plot_data = data_frame[important_features].dropna()

        sns.pairplot(plot_data, diag_kind='kde', plot_kws={'alpha': 0.5, 'color': 'blue'})
        plt.suptitle('Feature Relationships', fontsize=16)

        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while plotting: {e}")

# Tkinter GUI Setup
root = tk.Tk()
root.title("Improved Student Performance Prediction")

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

frame_train = ttk.Frame(notebook)
notebook.add(frame_train, text="Data Import & Train Model")
plot_frame = ttk.Frame(notebook)
notebook.add(plot_frame, text="Data Visualization")

frame_import = ttk.LabelFrame(frame_train, text="Import Data")
frame_import.pack(fill=tk.X, padx=10, pady=10)
frame_train_model = ttk.LabelFrame(frame_train, text="Train Model & Results")
frame_train_model.pack(fill=tk.X, padx=10, pady=10)

btn_import = ttk.Button(frame_import, text="Import CSV", command=import_csv)
btn_import.pack(pady=10)
btn_train = ttk.Button(frame_train_model, text="Train Improved Model", command=train_improved_model)
btn_train.pack(pady=10)
btn_plot = ttk.Button(plot_frame, text="Plot Improved Scatter", command=plot_improved_scatter)
btn_plot.pack(pady=10)

mse_text = tk.StringVar()
label_mse = tk.Label(frame_train_model, textvariable=mse_text, wraplength=400)
label_mse.pack(pady=5)

theta_text = tk.StringVar()
label_theta = tk.Label(frame_train_model, textvariable=theta_text, wraplength=400)
label_theta.pack(pady=5)

prediction_model_text = tk.StringVar()
label_model = tk.Label(frame_train_model, textvariable=prediction_model_text, wraplength=400)
label_model.pack(pady=5)

results_text = tk.StringVar()
label_results = tk.Label(frame_train_model, textvariable=results_text, wraplength=400)
label_results.pack(pady=5)

root.mainloop()
