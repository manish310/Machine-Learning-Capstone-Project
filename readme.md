#  TMDB Box-office Prediction
#### by Manish Kumar

## Input Files
The project contains two input files: _train.csv_ and _test.csv_ . The training dataset contains 3000 rows and 23 columns and testing dataset contains 4398 rows and 22 columns. Some of the important columns are as given below:
##### Important Features:
>  
  * __budget__ : *Continuous Numerical data.*
  * __genres__ : *Categorical data (in form of dictionary).*
  * __imdb_id__ : *Continuous Numerical data.*
  * __original_language__ : *Categorical data ('en', 'hi', 'ko', 'sr', 'fr', 'it', 'nl', 'zh', 'es', 'cs', 'ta',
       'cn', 'ru', 'tr', 'ja', 'fa', 'sv', 'de', 'te', 'pt', 'mr', 'da','fi', 'el', 'ur', 'he', 'no', 'ar', 'nb',
       'ro', 'vi', 'pl', 'hu','ml', 'bn', 'id').*
  * __original_title__ : *Textual data.*
  * __popularity__ : *Continuous Numerical data.*
  * __production_companies__ : *Categorical data.*
  * __production_countries__ : *Categorical data.*
  * __release_date__: *Date type.*
  * __runtime__ : *Continuous Numerical data.*
  * __spoken_languages__: *Categorical data.*
  * __status__ : *Categorical data ('Released', 'Rumored').*
  * __title__ : *Textual data.*
  * __cast__ : *Textual data.*
  * __crew__ : *Textual data.*
  * __revenue__ : *Continuous Numerical data.*

## Software Requirements:
To run this project successfully following software packages are required:

  * __Python version__: 2.7.1
  * __sklearn version__: 0.18.2
  * __Pandas version__: 0.20.3
  * __Numpy version__: 1.13.3
  * __Seaborn version__: 0.9.0
  * __Matplotlib version__: 2.2.3

## Python Codes (Jupyter Notebook):
The projects code are presented in Jupyter notebook file **TMDB_Capstone.ipynb**.

## Output Files:
The final output of the project is in form of _csv_ file: **output.csv**.
