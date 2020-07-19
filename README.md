# Hong-Kong Horse Race Betting Model

## Overview
The goal of this project was to develop a profitable betting model, focused on beating the highly efficient Hong-Kong horse race betting market. This was undertaken as a summer project to help exercise my predictive modeling skills. When I started this project, I focused on simple and obvious feautures such as past performances, best track types for each horse, and jockey and trainer skill. However, I soon realized that this would not be sufficent to beat the Hong-Kong markets. After more research, I stumbled upon a LinkedIn article by Charles Spenser, outlining his approach to beating the Singapore horse racing markets. After reaching out and numerous correspondances, I was motivated to aquire a copy of *Picking Winners* by Andrew Beyer. This text is commonly cited as essensial reading for anyone looking to begin serious, numerical-based horse handicapping. All speed rating calculations stem from the knowledge garnered throughout this text. 

## Data Collection
Data was sourced from *Horse Racing in HK*, a Kaggle dataset found at: https://www.kaggle.com/gdaley/hkracing/data. The raw data contains two files, races.csv and runs.csv. races.csv contains information regarding the overall race, such as date, venue, distance, winnings, etc. runs.csv focuses on the specific runs of each horse, and contains data such as sectional times and horse information. These datasets were merged together on the race_id column to create a comprehensive data file. 

## Data Cleaning
Due to the refined nature of the downloaded dataset from Kaggle, very little data cleaning was required. The main goal of this phase of the project was to remove outlying data that would impact speed rating calculations and outlier target values. This was performed in /Data/CleanData.ipynb. 

The first step was to count the various occuracnes of the different race catagories accounted for in the speed rating calculations: venue, distance, and class. 


## Feature Engineering

## Preditive Model

## Betting Model

## Results

## Reflection and Next Steps

