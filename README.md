# Student Score Prediction

## Overview

This project uses various regression models to predict student scores using the dataset: [Student Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977) from Kaggle. The best-performing model is deployed as a Flask web application on [Render](https://mlproject-hw1j.onrender.com/predictdata), for on-demand predictions with new data. Exploratory Data Analysis (EDA) reveals that Student's Performance is related to lunch, race, and parental level of education; females have a higher pass percentage and are top scorers; Student's Performance has a limited correlation with test preparation course completion, though completing the preparation course is generally beneficial.


## Project Components

The project consists of the following main components:

* **Data Ingestion**: Handles loading the dataset and splitting it into training and testing sets.
* **Data Transformation**: Preprocesses the data, including handling categorical and numerical features.
* **Model Trainer**: Trains a machine learning model to predict student scores.
* **Prediction Pipeline**:  A separate pipeline to use the trained model for making predictions on new data.
* **Utility Functions**:  Provides helper functions for tasks like saving and loading objects.
* **Flask Application**: A Flask-based web application for serving predictions.

###   1. Data Ingestion

The `DataIngestion` component is responsible for:

* Loading the student exam performance dataset (from a CSV file).
* Splitting the data into training and testing sets.
* Saving the training and testing sets as CSV files.

 
### 2. Data Transformation

The `DataTransformation` component handles:

* Preprocessing the data for model training.
* This includes:
    * Imputing missing values.
    * Scaling numerical features.
    * One-hot encoding categorical features.
* Creating a preprocessing pipeline using `scikit-learn`.
* Saving the preprocessor object for later use in prediction.


###   3. Model Trainer

The `ModelTrainer` component is responsible for:

* Training a machine learning model to predict student scores.
* Evaluating the model's performance.
* Saving the trained model.

### 4. Prediction Pipeline

The `PredictPipeline` component is designed to:
* Load the trained model and preprocessor.
* Preprocess new input data.
* Use the model to predict student scores.


### 5. Utility Functions

The `src.utils` module provides helper functions:

* `save_object()`:  Saves Python objects (like the preprocessor or trained model) using `dill`.
* `load_object()`: Loads saved Python objects.
* `evaluate_model()`:  Trains and evaluates multiple models, performs hyperparameter tuning with `GridSearchCV`, and returns performance metrics.

### 6. Flask Application

The `app.py` file contains a Flask application that:

* Serves a web page (`home.html`) for users to input student data.
* Uses the `PredictPipeline` to predict the student's math score based on the input.
* Displays the prediction to the user.
* Includes a basic home page (`index.html`).


##   Environment Setup

To run this project (for demonstration purposes), you'll need to set up a Python environment and install the required dependencies.

1.  **Python**:  Ensure you have Python 3.12 installed.
2.  **Virtual Environment**:  It's recommended to use a virtual environment:

    ```bash
    conda create -n venv python=3.12
conda activate venv
    ```
3.  **Dependencies**:  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file includes packages like:

    * `scikit-learn`
    * `pandas`
    * `numpy`
    * `flask`
    * `catboost`
    * `xgboost`
    * `dill`

4.  **Running the Flask Application**:
    ```bash
    python app.py
    ```
    The application will typically run on `http://127.0.0.1:5000/`.

##   Project Structure

The repository has the following structure:

* `notebook/`: Contains the notebook with the EDA and initial model experimentation
* `src/`: Contains the source code for the project, including:
    * `components/`:  Modules for data ingestion, transformation, and model training.
    * `pipeline/`:  Modules for the prediction pipeline.
    * `utils.py`:  Utility functions.
    * `exception.py`:  Custom Exception class.
    * `logger.py`:  Custom Logger.
* `templates/`: Contains the HTML templates for the Flask application.
* `app.py`:  Runs the Flask application.
* `requirements.txt`: Lists the project dependencies.
* `setup.py`:  A setup script.
