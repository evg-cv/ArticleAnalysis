# ArticleAnalysis

## Overview

This project is to estimate the category of the sentences of the text and the similarity between each sentence & title.
For this, the main 2 models are trained with the training data - the model for sentence category and the model for 
similarity. The Gensim, NLTK, Spacy libraries are used in this project and also, a pre-trained model which is part of a 
model that is trained on 100 billion words from the Google News Dataset is used for feature extraction.

## Structure

- src

    The main source code for NLP

- utils

    * The models for this project
    * The source code for the management of the folders and files

- app

    The main execution file

- trainer

    The execution file for training

- title

    The execution file for title similarity estimation
    
- requirements

    All the dependencies for this project
    
- Settings

    The settings for file path

## Installation

- Environment

    Ubuntu 18.04, Windows 10, Python 3.6,
     
- Dependency Installation

    Please go ahead to this project directory and run the following commands in the terminal
    ```
        pip3 install -r requirements.txt
        python3 -m spacy download en_core_web_sm
        python3 -m nltk.downloader all
    ```

- Please create the "model" folder in "utils" folder and copy the word model to "model" folder.

## Execution

- For training the model, please set TRAINING_DATA_PATH variable in settings file with the absolute path of the training 
data file and run the following command in the terminal.

    ```
        python3 trainer.py
    ```

- After training, to get the result for the article, please set INPUT_EXCEL_PATH variable in settings file with the 
absolute path of the excel file and run the following command in the terminal.

    ```
        python3 app.py
    ```

- To compare the origin title with other titles, please set TITLE_SIMILARITY_EXCEL_PATH variable in settings file with 
the absolute path of the excel path for title similarity estimation and run the following command.

    ```
        python3 title.py
    ```
  
## Note

- When creating the training data file, input file, title file as an input file, please make the fields of the files the same as 
the sample file.

- When inputting the values into the LABEL field of the training data, if the sentence is non-pertinent to title, you don't 
have to insert any value like the part marked with red in the sample training data file.

- When making title file, the other titles in OTHER TITLES field has to be separated with ";".

- All of training data, input file and title file has to be excel format.
