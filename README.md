# Predicting Hospital Readmissions Rates

![Project Logo](path/to/logo.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Objectives](#objectives)
- [Prerequisites and System Requirements](#prerequisites-and-system-requirements)
- [Methodology](#methodology)
    - [Data Cleaning](#data-cleaning)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Feature Importance](#feature-importance)
    - [Model Training and Evaluation](#model-training-and-evaluation)
    - [Cross-Validation](#cross-validation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Examples and Use-Cases](#examples-and-use-cases)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Diagrams and Flowcharts](#diagrams-and-flowcharts)
- [Contact](#contact)

## Project Overview
This project aims to predict hospital readmissions using a dataset that includes various estimates related to hospital performance. The objective is to develop a machine learning model that can accurately predict the likelihood of patient readmissions, helping healthcare providers improve patient outcomes and reduce costs.

## Dataset Description
- **Source**: [Kaggle] (https://www.kaggle.com/datasets/thedevastator/us-healthcare-readmissions-and-mortality)
- **Number of Records**: [64764]
- **Features**:
  - **Denominator**: Total number of patients
  - **Lower Estimate**: Lower bound estimate of readmission
  - **Higher Estimate**: Upper bound estimate of readmission
  - **Score**: Actual readmission score

## Objectives
- Predict hospital readmissions using machine learning
- Identify key predictors of readmission
- Provide actionable insights to healthcare providers

## Prerequisites and System Requirements
- **Python 3.6+**
- **Libraries**:
  - pandas
  - NumPy
  - matplotlib
  - seaborn
  - Scikit-learn
  - joblib

To install the required libraries, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Methodology

### Data Cleaning
1. Handled missing values by dropping rows with NaNs.
2. Converted data types to ensure consistency.
3. Normalised features to prepare the data for modelling.

### Exploratory Data Analysis (EDA)
1. Examined the distribution of scores using histograms.
2. Performed correlation analysis and visualised it using heatmaps.

### Feature Importance
1. Trained a Random Forest model.
2. Identified 'Lower Estimate' and 'Higher Estimate' as significant predictors using feature importance plots.

### Model Training and Evaluation
1. **Model Used**: Random Forest Classifier
2. **Accuracy**: 99.98%
3. **Metrics**: Precision, Recall, F1-Score, Confusion Matrix

### Cross-Validation
1. Performed 5-fold cross-validation.
2. **Mean Accuracy**: 95.84%
3. **Standard Deviation**: 0.0012

## Results
- **Key Findings**:
  - 'Lower Estimate' and 'Higher Estimate' are critical predictors.
  - High model accuracy and stability suggest reliable performance.

## Conclusion
The project successfully developed a machine-learning model that accurately predicts hospital readmissions. Healthcare providers can use this model to identify high-risk patients and take preventive measures.

## Future Work
- Explore additional features to improve the model.
- Experiment with different modelling techniques.
- Deploy the model in a real-world healthcare setting.

## Examples and Use-Cases
Example 1. **Healthcare Providers**: Use the model to identify patients at high risk of readmission and implement targeted interventions.
  Input: Patient data including Denominator, Lower Estimate, and Higher Estimate.
  Output: Probability of readmission.
Example 2. **Policy Makers**: Allocate resources more effectively by using the model to predict readmission rates and understanding readmission trends.
Example 3. **Researchers**: Explore the dataset to find new insights and improve healthcare outcomes.

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Required Python libraries (listed in `requirements.txt`)

## Reproducing the Analysis
1. **Clone the repository**:
   
```bash
git clone https://github.com/Mariejlo/Data-analytics-projects-.git
 ```
2. Navigate to the project directory:
    ```sh
    cd Data-analytics-projects-
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
## Usage

### Example Command
```sh
python main.py --example-argument
```
4. Open the Jupyter Notebook or your preferred IDE and run the code blocks step-by-step.

### Example Code
```python
# Example code snippet
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('path/to/dataset.csv')

# Perform some analysis
df.describe()
```

## Diagrams and Flowcharts

### Project Workflow
![Project Workflow](path/to/workflow.png)

### Data Processing Flowchart
![Data Processing Flowchart](path/to/data_processing_flowchart.png)

### Model Training Pipeline
![Model Training Pipeline](path/to/model_training_pipeline.png)

## Methodology

The methodology for this project involved several key steps:

1. **Data Collection**: The dataset was collected from [https://www.kaggle.com/datasets/thedevastator/us-healthcare-readmissions-and-mortality].
2. **Data Preprocessing**: Steps included handling missing values, encoding categorical variables, and scaling features.
3. **Exploratory Data Analysis (EDA)**: Visualizations and statistical analyses were performed to understand the data.
4. **Feature Engineering**: New features were created to enhance model performance.
5. **Model Development**: Various models were developed and evaluated to select the best-performing one.
6. **Model Evaluation**: The selected model was evaluated using accuracy, precision, recall, and F1-score metrics.
7. **Model Validation**: Cross-validation was performed to validate the robustness of the model.

## Results

The results of the analysis and model evaluation are as follows:

- **Accuracy**: 0.99976
- **Confusion Matrix**:
    ```
    [[8330    0]
     [   2   25]]
    ```
- **Classification Report**:
    ```
                  precision    recall  f1-score   support

             low       1.00      1.00      1.00      8330
          medium       1.00      0.93      0.96        27

        accuracy                           1.00      8357
       macro avg       1.00      0.96      0.98      8357
    weighted avg       1.00      1.00      1.00      8357
    ```

## Conclusion

- The Random Forest model achieved high accuracy, indicating excellent performance in predicting readmission probabilities.
- Key features contributing to the model's performance include the Denominator, Lower Estimate, and Higher Estimate.

## Future Work

- Explore additional features and data sources to enhance the model's performance.
- Implement the model in real-world healthcare to monitor and predict patient readmissions.
- Develop a user-friendly interface for healthcare professionals to interact with the model.

## Examples and Use-Cases

### Example 1: Predicting Patient Readmission
- Input: Patient data includes denominator, lower estimate, and higher estimate.
- Output: Probability of readmission.

### Example 2: Healthcare Resource Allocation
- Use the model to predict readmission rates and allocate resources accordingly.

### **Contact / Support**
For support or inquiries, feel free to contact [Marie.lopator@gmail.com]

Best of luck!
