## Disaster Response Pipeline Project
This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set used in this project contains real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that the messages can be sent to the appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app displays visualizations of the data as well.

## Table of Contents
- [Installation](#installation)
- [File Description](#file-description)
- [Instructions to Run the Project](#instructions-to-run-the-project)
- [Results](#results)
- [Acknowledgement](#acknowledgement)

## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.x.

See `requirements.txt`.

## File Description
This repo contains 3 folders which includes work related to data transformation, data modeling and app development. Here's the file structure of the project:

    .
    ├── app
    ├   ├── templates
    ├   ├   ├── go.html               # classification result page of web app
    ├   ├   ├── master.html           # main page of web app
    ├   ├── run.py                    # Flask file that runs app
    ├── data 
    ├   ├── disaster_categories.csv   # data to process
    ├   ├── disaster_messages.csv     # data to process
    ├   ├── DisasterResponse.db       # saved database
    ├   ├── process_data.py           # script to run ETL pipeline
    ├── model
    ├   ├── train_classifier.py       # script to run ML pipeline
    ├   ├── classifier.pkl            # saved model
    ├── Requirements.txt
    └── README.md

## Instructions to Run The Project
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results
The main findings, which is the web app can be found by running python scripts.

## Acknowledgement
Must give credit to [Figure Eight](https://www.figure-eight.com/) for the data.
