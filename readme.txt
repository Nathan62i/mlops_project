# Sentiment Analyzer Project  

## Project Description  

This project aims to develop a complete application to analyze sentiments expressed in textual reviews (positive or negative evaluations).  

The main objectives are:  
- Build, train, and track machine learning models using **MLFlow** to ensure reproducibility and experiment tracking.  
- Implement a REST web application via **FastAPI** to predict sentiments from textual reviews.  
- Deploy the application as a Docker container for portable usage.  

---

## Project Features  

- **Experiment Tracking**:  
  - Use MLFlow to log parameters, metrics, models, and other data related to experiments.  
  - Manage a model registry for deployment purposes.  

- **Web Application**:  
  - REST endpoint for submitting textual reviews and receiving sentiment analysis results.  
  - Automatic documentation via FastAPI.  

- **Docker Deployment**:  
  - Containerize the application for easier execution and sharing.  

---

## Project Structure  

Below is the project structure and the main files:  

├── docker-compose.yml
├── mlruns
│   ├── 0
│   │   └── meta.yaml
│   └── models
├── notebooks
│   ├── mlartifacts
│   ├── mlruns
│   │   └── models
│   │       └── logisticRegression
│   │           ├── meta.yaml
│   ├── model_design_2.ipynb
│   ├── model_design_3.ipynb
│   ├── model_design.ipynb
│   └── test_model.ipynb
├── readme.txt
├── requirements.txt
├── setup.py
├── src
│   ├── frontend
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── sentiment_analyzer
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   ├── predict.py
│   │   ├── promote.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── model_manager.cpython-310.pyc
│   │   │   └── predict.cpython-310.pyc
│   │   └── tests
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   └── test_model.cpython-310-pytest-8.3.3.pyc
│   │       └── test_model.py
│   └── sentiment_analyzer.egg-info
│       ├── dependency_links.txt
│       ├── entry_points.txt
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       └── top_level.txt
└── webapp
    ├── app.py
    ├── Dockerfile
    ├── get_mlflow_model.py
    ├── __pycache__
    │   └── app.cpython-310.pyc
    └── sentiment-analyzer-model
        ├── conda.yaml
        ├── MLmodel
        ├── model.pkl
        ├── python_env.yaml
        └── requirements.txt


---

## Main Files Overview  

- **src/sentiment_analyzer/model_design_2.ipynb**:  
  - Contains the notebook to define and train a machine learning model.  
  - Integrated with MLFlow to track experiments.  

- **src/sentiment_analyzer/get_mlflow_model.py**:  
  - Loads a model from the MLFlow registry for local or application use.  

- **tests/test_model.py**:  
  - Includes unit and functional tests to ensure model performance and robustness.  

- **webapp/app.py**:  
  - Implements the web application with FastAPI.  
  - Provides sentiment predictions based on submitted reviews via a REST API.  

- **webapp/Dockerfile**:  
  - Describes the steps to build a runnable Docker image of the web application.  

- **setup.py**:  
  - Allows the project to be packaged as an installable Python package.  

- **requirements.txt**:  
  - Lists all necessary dependencies to run the project.  

---

## Basic Instructions  

### 1. Install Dependencies  

Create a Python environment and install the dependencies:  
```bash
pip install -r requirements.txt

### 2. Start the MLFlow Server  
Launch a local MLFlow server to track your experiments:  
```bash
mlflow server

### 3. Train a Model  
Open the `model_design_2.ipynb` notebook and execute it to train a model while logging experiments to MLFlow.  

### 4. Launch the Web Application  
Load a model from MLFlow and start the FastAPI server:  
```bash
uvicorn webapp.app:app --host 0.0.0.0 --port 8000

### 5. Run Tests  
Validate the model functionalities:  
```bash
pytest tests/

### 6. Build a Docker Image  
From the `webapp/` directory, build the Docker image:  
```bash
docker build -t sentiment-analyzer .

docker run -p 8000:8000 sentiment-analyzer

### Additionnal documentation 

- MLFlow Documentation
- FastAPI Documentation
- Docker Documentation
