# Forecaster App

The **Forecaster App** is designed to provide sales forecasts for specific stores and items over a selected time period. The app leverages the XGBoost forecasting method for accurate predictions and analysis. Below is an overview of the app's features and functionality.

## Features

### Sidebar Outputs
- **Predicted Sales Summary**: Displays the total predicted sales for each month (up to the selected month) for the chosen store and item.

### Line Chart
- Visualizes the predicted sales trends for the first three months (January to March 2014) for the selected store and item.

### Tabular Data
- Displays detailed prediction data (date, month name, and predicted sales) for the selected month.

### Error/Warning Messages
- Provides appropriate warnings or error messages if no data is available for the selected store, item, or month.

---

## Model Training

The XGBoost model was trained using Jupyter Notebook in VS Code. A detailed description of the model development process, along with the code, is available in the folder `Used Notebooks For Time Series Analysis` under the file name `Hyperparameter_Tuning.ipynb`. Additional notebooks used to prepare `Hyperparameter_Tuning.ipynb` are also included in this folder.

---

## Reproducing the Model

If you wish to reproduce the model in Google Colab, follow these steps:

1. Download the `train_final.csv` file from [this link](https://drive.google.com/file/d/17PLomDukjPaGJLj7hipzSyJOQMNTMwvj/view?usp=drive_link).
2. Upload the file to your Colab environment.
3. Run the provided notebooks to reproduce the model.

Alternatively, you can test the app locally by following these steps:

1. Clone the repository and create a virtual environment in Python.
2. Install all the required libraries using the `requirements.txt` file.
3. Run the following command:
   ```bash
   streamlit run app.py
   ```
4. This will open a webpage where you can forecast predictions for the first three months of 2014.
5. Note: The `train_final.csv` file is large (905.9 MB) and will be downloaded into the project's `data` folder from Google Drive. The process may take a few minutes.

---

## Requirements

To run this project locally, ensure you have the following dependencies installed (as listed in `requirements.txt`):
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- XGBoost
- Streamlit
- And more...

For the full list of dependencies, refer to the `requirements.txt` file.

---

## License

This project is licensed under the [MIT License](LICENSE).
