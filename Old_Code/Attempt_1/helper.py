import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

# Create our evaluate function
def winnerEval(model, x_test, y_test, target, race_sizes):
    # convert preds into an actual win choice
    winPreds = model.predict_proba(x_test)[:, 1]
    winCount = 0
    temp = 0 
    for i, s in enumerate(race_sizes):
        low_index = temp
        high_index = temp + s
        
        racePreds = winPreds[low_index:high_index]
        raceVals = y_test[low_index:high_index]
        
        if target=='won':
            predWinner = np.argmax(racePreds, axis=0)
            actWinner = np.argmax(raceVals, axis=0)
        
            if predWinner == actWinner:
                winCount += 1
                
        elif target=='placed':
            predPlacers = racePreds.argsort()[-3:]
            actPlacers = raceVals.argsort()[-3:]
            
            for val in actPlacers:
                if val in predPlacers:
                    winCount+=1
            
        temp += s
        
    if target=='won':
        return winCount/float(len(race_sizes))
    else:
        return winCount / float(len(race_sizes)*3)

def crossVal(dat, feat, target, model, skipPct=0, n_folds=4):
    
    dat = dat.copy()
    race_ids = np.unique(dat["race_id"])[int(len(np.unique(dat["race_id"]))*skipPct):]
    cv = KFold(n_splits=n_folds, shuffle=True)
    
    scores = []
    count = 1
    for train_races, test_races in cv.split(race_ids):
        print("CV {}/{}".format(count, n_folds))
        X_train = dat.loc[dat["race_id"].isin(train_races)][feat]
        y_train = dat.loc[dat["race_id"].isin(train_races)][target]
        X_test = dat.loc[dat["race_id"].isin(test_races)][feat]
        y_test = dat.loc[dat["race_id"].isin(test_races)][target]
        
        model.fit(X_train, y_train)
        
        testDat = dat.loc[dat["race_id"].isin(test_races)]
        testGroups = [len(testDat.loc[testDat["race_id"]==race_id]) for race_id in np.unique(testDat["race_id"])]
        scores.append(winnerEval(model, X_test, y_test, target, testGroups))
        count += 1
        
    meanScore = np.mean(scores)
    stdScore = np.std(scores)
    
    print("Mean score: {:.3f} +/- {:.3f}".format(meanScore, stdScore))
    
    return meanScore, stdScore

# DO ANOTHER FUNCTION THAT IS JUST FINDING THE BEST ODDS HORSE
def bestOddsEval(data, TARGET):
    
    race_sizes = [len(data.loc[data["race_id"]==race_id]["race_id"]) for race_id in np.unique(data["race_id"])]
    winCount = 0
    temp = 0 
    for i, s in enumerate(race_sizes):
        low_index = temp
        high_index = temp + s
        
        oddsPreds = data[["win_odds", "place_odds"]][low_index:high_index]
        raceVals = data[TARGET][low_index:high_index]
        
        if TARGET=='won':
            predWinner = np.argmin(oddsPreds["win_odds"], axis=0)
            actWinner = np.argmax(raceVals, axis=0)

            if predWinner == actWinner:
                winCount += 1
            
        elif TARGET=='placed':
            predPlacers = oddsPreds["place_odds"].argsort()[:3].to_list()
            actPlacers = raceVals.argsort()[-3:].to_list()
            print(predPlacers, actPlacers)
            
            for val in actPlacers:
                if val in predPlacers:
                    winCount+=1
                    
        temp += s
        
    if TARGET=='won':
        return winCount/float(len(race_sizes))
    else:
        return winCount / float(len(race_sizes)*3)
    
def selectFeatures_k(in_model, data, features, target, testPct=0.2):
    # Perform feature selection with select_from_model
    X = data[features]
    y = data[target]

    trainIndex = int(len(np.unique(data["race_id"])) * (1-testPct))
    max_race_id = np.unique(data["race_id"])[trainIndex]
    X_train = X.loc[data["race_id"]<=max_race_id]
    y_train = y.loc[data["race_id"]<=max_race_id]
    X_test = X.loc[data["race_id"]>max_race_id]
    y_test = y.loc[data["race_id"]>max_race_id]

    race_sizes_for_eval = [len(data.loc[data["race_id"]==race_id]["race_id"]) for race_id in np.unique(data.loc[data["race_id"]>max_race_id]["race_id"])]

    Ks = np.linspace(2, len(features), len(features)-1, dtype=int)
    scores = []
    feats = []
    for k in Ks:
        print("k: {}/{}".format(k, max(Ks)))
        pipe = Pipeline([('skb', SelectKBest(chi2, k=k)),
                            ('model', clone(in_model))])
        pipe.fit(X_train, y_train)
        
        acc = winnerEval(pipe, X_test, y_test, target, race_sizes_for_eval)
        scores.append(acc)
        
        featuresUsed = []
        for i, col in enumerate(X_train.columns):
            if pipe["skb"].get_support()[i]:
                featuresUsed.append(col)
                
        feats.append(featuresUsed)
        
    return scores, feats