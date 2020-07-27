# Hong-Kong Horse Race Betting Model

## Overview
The goal of this project was to develop a profitable betting model, focused on beating the highly efficient Hong-Kong horse race betting market. This was undertaken as a summer project to help exercise my predictive modeling skills. When I started this project, I focused on simple and obvious feautures such as past performances, best track types for each horse, and jockey and trainer skill. However, I soon realized that this would not be sufficent to beat the Hong-Kong markets. After more research, I stumbled upon a LinkedIn article by Charles Spenser, outlining his approach to beating the Singapore horse racing markets. After reaching out and numerous correspondances, I was motivated to aquire a copy of *Picking Winners* by Andrew Beyer. This text is commonly cited as essensial reading for anyone looking to begin serious, numerical-based horse handicapping. All speed rating calculations stem from the knowledge garnered throughout this text. 

## Data Collection
Data was sourced from *Horse Racing in HK*, a Kaggle dataset found at: https://www.kaggle.com/gdaley/hkracing/data. The raw data contains two files, races.csv and runs.csv. races.csv contains information regarding the overall race, such as date, venue, distance, winnings, etc. runs.csv focuses on the specific runs of each horse, and contains data such as sectional times and horse information. These datasets were merged together on the race_id column to create a comprehensive data file. 

## Data Cleaning
Due to the refined nature of the downloaded dataset from Kaggle, very little data cleaning was required. The main goal of this phase of the project was to remove outlying data that would impact speed rating calculations and outlier target values. This was performed in /Data/CleanData.ipynb. 

The first step was to count the various occuracnes of the different race catagories accounted for in the speed rating calculations: venue, distance, and class. 

![alt text](https://github.com/NoahRowe/HorseBetting/blob/master/Data/venue_counts.png?raw=true)
![alt text](https://github.com/NoahRowe/HorseBetting/blob/master/Data/class_counts.png?raw=true)
![alt text](https://github.com/NoahRowe/HorseBetting/blob/master/Data/distance_counts.png?raw=true)

Races were removed from the dataset based on the frequency of that type of race. For example, races at HV at a distance of 2200 were very uncommon, so they were excluded. Also, as one of my predictive models used lengths behind as a target, this feature had to be cleaned. The issue was that if a horse did not finish a race, it was assigned a final lengths behind value of 999. For these runs, lenghts behind was replaced with the average lengths behind of all last place horses. 

## Feature Engineering
Many features were experimented with throughout the course of this model. See the **adding_features** directory for the full code. The final features chosen for the model are as follows:
* gear change: 1 if the horse is running with different gear than its last race, 0 if not
* horse rating: the rating assiged to the horse by the Hong-Kong Jockey Club, relative to other horses in the race
* last speed rating: the calculated speed rating for the horses last run, relative to other horses in the race
* average speed rating: avgerage calculated speed rating for this horses past runs, relative to other horses in the race
* best surface distance: the best speed rating acheived by this horse on this surface and at this distance, relative to other horses in the race
* weight rating: predicted speed rating based on running weight, derrived from a linear regression based on previous speed ratings and weight values, relative to other horses in the race
* rest rating: predicted speed rating based on time since last run, derrived from a linear regression based on previous speed ratings and rest values. 
* going rating: average speed rating on this going type, relative to other horses in the race
* jockey, trainer, horse win_percent: percent of races won by jockey, trainer, and horse, relative to other runners in the race
* jockey, trainer, horse normalized record: average of past finishes divided by number of horses in each race, relative to other horses in the race
* trainer-jockey recored: record with this combination of trainer and jockey, relative to other horses in the race
* draw win percent: the win percentage of horses running at this post position
* average section 1 speed rating: the average speed rating for the first section of the race for this horse, relative to other horses in the race
* last section 1 speed rating: the speed rating for this horses last run over the first section of the race, relative to other horses in the race
* average last section speed rating: the average speed rating for this horse over the last section of the race, relative to other horses in the race
* last last section speed rating: the speed rating for this horse over the last section of its most recent race, relative to other horses in the race
* average speed rating ratio,: the average of the ratio of first section speed rating to last section speed rating for each race, relative to other horses in the race
* last speed rating ratio: the ratio of first section speed rating to last section speed rating for this horses most recent run, relative to other horses in the race

Most features were scaled relative to the statistics of the other horses in this race. Scaling was done using sklearn.preprocessing.StandardScalar (this was found to return better results than min-max or robust scaling functions). 

## Preditive Model
Initially, many different machine learning models were experimented with the goal of finding the best one to improve results. However, it was soon discovered that the best way to boost results was to focus on data cleaning, feature engineering, and target selection. Based on the Charles Spenser article mentioned above, I attempted to determine the relationship between lengths behind winner and implied probability of winning. I was looking to use a regression model, rather than classification, due to the extra information that could be gleaned from a continious variable output. However, I soon came to realize that a more simplistic solution would be to use the predict_proba method from the logistic regression model. This allowed for the prediction of a discrete variable (won or not) and to see the predicted probability that the prediction is true. A logisitc regression model was also useful due to its speed and the easy comprehension associated with having won or not as the target variable.

## Betting Model
Various betting models were tested and tried, most of which were centered around Kelly's Criterion. Kelly's Criterion (https://www.investopedia.com/terms/k/kellycriterion.asp) is a bankroll management strategy that implies what bets should be taken. The formula requires knowlege of the implied probabilities from the set odds on each horse and the model output chance of winning. By comapring the payout and the 'known odds' (from the model), the formula will assess the risk and reward of the bet to advise how much of the capital to bet. Kelly's Criterion can be modified by betting a fraction of the advised ammount to reduce risk. 

A useful way to account for odds in a betting model is to filter the bets you should take based on where your model performs best. My model was better at predicting win percentages for horses with decimal odds between 4 - 8. Therefore, my model could be improved considerably by only betting on these horses. When doing this, it was important to look out for data leakage. It would be poor data science if the final model was ran on data that we used to determine where the model was successful. 

## Results
In the end I was unable to generate positive returns from my model. I think that this is due to the exceptional efficency of the Hong-Kong horse racing market, and believe that I would have had more success in a less intense market. The image below is a cross validation run of 6 models on my testing dataset, using a odds-based filter (mentioned above) and Kelly's Criterion. 


## Reflection and Next Steps

