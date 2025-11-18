# A-new-enhanced-machine-learning-plagioclase-liquid-hygrometer
An enhanced machine learning plagioclase-liquid hygrometer for magmatic H2O estimation across different tectonic settings
## Table of Contents
<ul>
<li>Requirements
<li>Installation
<li>Usage
<ul>
<li>(with T&P) plagioclase-liquid hygrometer.py</li>
<li>(without T&P) plagioclase-liquid hygrometer.py</li>
<li>Data Enhancement_MCMC.py</li>
</ul>
</li>
<li>Data Requirements</li>
<li>Output Files</li>
<li>Troubleshooting</li>
<li>License</li>
</ul>

## Requirements

To successfully run these models, you need to have Python 3.11 or above installed on your system.

Required Python Packages
<ul>
<li>Python 3.11+
<li>Spyder (or any other compiler that can run python)
</li> 
</ul>

Required Python libraries:
<ul>
<li>pandas
<li>numpy
<li>scikit-learn
<li>pymc(version=5.18.2)
<li>pytensor(version=2.26.3)
<li>arviz(version=0.20.0)
  </li>
</ul>

### To install these dependencies, you can create a virtual environment, which we recommend using anaconda3 to do:

#### Create a virtual environment (optional but recommended)
<ul>
  <li>In a windows cmd window enter the command: conda create -n your_env_name python=3.11</li>
</ul>

#### Install required packages
<ul>
 <li>pip install pandas numpy scikit-learn pymc pytensor arviz</li>
</ul>

#### The installed anaconda3 usually comes with the spyder compiler, in the cmd window enter the following command to switch the virtual environment and open the spyder:

<ul>
  <li>conda activate your_env_name</li>
  <li>spyder</li>
</ul>

### Additional Data Requirements
<ul>
  <li>The hygrometer require datasets in .xlsx format. Ensure that your data includes the relevant columns, such as geochemical elements, temperature and pressure.</li>
</ul>

## Repository Contents
<ol>
<li>plagioclase-liquid hygrometer.py: </li>
<p>This code builds a machine learning hygrometer to predict magmatic H2O content in different tectonic settings using geochemical elemental data and thermodynamic parameters.</p>
<li>Data Enhancement_MCMC.py:</li> 
<p>This code augments the data using the Markov chain Monte Carlo method.</p>
</ol>

## Instructions
### 1. (with T&P) plagioclase-liquid hygrometer.py
This code is designed to predict magmatic H2O content in different tectonic settings from geochemical measurements using geochemical elemental data and thermodynamic parameters. Follow the steps below to run the code successfully:

#### Prepare the Dataset: 
<p>Ensure you have a .xlsx file with geochemical data of plagioclase and silicate melts in equilibrium with it and temperature and pressure for magmatic H2O content estimation. The geochemical data should include various major elements (such as SiO₂, TiO₂, Al₂O₃, etc.) with proper order (see Table S1) as input features and a column labeled H2O for the target variable. 

#### Input File Path: 
<p>You need not make any alterations to the input path. Simply enter the required data into the corresponding cells of the attached Excel file ‘Input_Lin2025’ (remember to enter zeros for any missing data). The code will automatically perform the calculations and output the results.</p>
  
#### Run the Code: The code will:
<ul>
  <li>Load the dataset and extract features (geochemical elements) and the target (H2O).</li>
  <li>Train a hygrometer model to predict magmatic H2O content based on the geochemical inputs.</li>
</ul>

#### Optional: 
You can modify the machine learning workflow, such as performing hyperparameter tuning or cross-validation to optimize the model further.

#### Results: 
Once the model is trained, it can be used to predict magmatic H2O content in different tectonic settings. You can save the trained model and use it in future predictions.

## Instructions
### 2. (without T&P) plagioclase-liquid hygrometer.py
This code is designed to predict magmatic H2O content in different tectonic settings from geochemical measurements using geochemical elemental data and thermodynamic parameters. Follow the steps below to run the code successfully:

#### Prepare the Dataset: 
<p>Ensure you have a .xlsx file with geochemical data of plagioclase and silicate melts in equilibrium with it and temperature and pressure for magmatic H2O content estimation. The geochemical data should include various major elements (such as SiO₂, TiO₂, Al₂O₃, etc.) with proper order (see Table S1) as input features and a column labeled H2O for the target variable. 

#### Input File Path: 
<p>You need not make any alterations to the input path. Simply enter the required data into the corresponding cells of the attached Excel file ‘Input_Lin2025’ (remember to enter zeros for any missing data). The code will automatically perform the calculations and output the results.</p>
  
#### Run the Code: The code will:
<ul>
  <li>Load the dataset and extract features (geochemical elements) and the target (H2O).</li>
  <li>Train a hygrometer model to predict magmatic H2O content based on the geochemical inputs.</li>
</ul>

#### Optional: 
You can modify the machine learning workflow, such as performing hyperparameter tuning or cross-validation to optimize the model further.

#### Results: 
Once the model is trained, it can be used to predict magmatic H2O content in different tectonic settings. You can save the trained model and use it in future predictions.

### 3. Data Enhancement_MCMC.py
<p>This code focuses on augmenting the data using the Markov chain Monte Carlo method. </p>

Steps to Use:
#### Prepare the Dataset: 
<p>Ensure that your dataset includes columns for geochemical data of plagioclase and silicate melts in equilibrium with it, 
 errors in EMPA of geochemical data, temperature and pressure.</p>

#### Edit File Path: 
Update the file path in the code to point to your dataset. dataFile = '/path/to/your/dataset.xlsx'

#### Customize Models: 
You can adjust the Data Expansion Multipliers, Number of Markov chains on your preferences to find higher quality extended datasets.

#### Save the Output: 
The script saves the augmented result as a table. Make sure to specify your desired file path in the code where the table will be saved.

### Usage Notes
#### Running the Code
To run the codes:
<ol>
<li>Clone this repository or download the codes.</li>
<li>Navigate to the directory where the codes are stored.</li>
<li>Launch Your virtual environment(if there is) and spyder(or any other compiler that can run python): </li>
<li>Open either the plagioclase-liquid hygrometer.py or Data Enhancement_MCMC.py code and run the cells in order.</li>
</ol>

### Output Files
<ol>
<li>plagioclase-liquid hygrometer.py: The output is a machine learning model trained to predict magmatic H2O content based on geochemical data and thermodynamic parameter. You can use this model to make predictions on new data.</li>
<li>Data Enhancement_MCMC.py: The output is a table file containing initial data and augmented data.</li>
</ol>

### File Paths
Ensure that you update the file paths for your datasets and output files as needed in each code. 

## Troubleshooting
<ul>
<li>Ensure that your data is correctly formatted and that there are no missing values in critical columns. If there are missing values, fill them all in with 0</li>
<li>Check that the Python environment has the correct versions of the required libraries installed.</li>
</ul>
