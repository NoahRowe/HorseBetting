import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
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
        
        if target=="HorseWin":
            predWinner = np.argmax(racePreds, axis=0)
            actWinner = np.argmax(raceVals, axis=0)

            if predWinner == actWinner:
                winCount += 1
            
        elif target=='HorseRankTop3':
            predPlacers = racePreds.argsort()[-3:]
            actPlacers = raceVals.argsort()[-3:]
            
            for val in actPlacers:
                if val in predPlacers:
                    winCount+=1
                  
        else:
            tc = 0 
            half = int(len(racePreds)/2.)
            predPlacers = racePreds.argsort()[-half:]
            actPlacers = raceVals.argsort()[-half:]
            
            for val in actPlacers:
                if val in predPlacers:
                    tc+=1
                    
            if tc!=0:
                winCount += tc/float(half)
                    
        temp += s
        
    if target=='HorseWin':
        return winCount/float(len(race_sizes))
    elif target=="HorseRankTop3":
        return winCount / float(len(race_sizes)*3)
    else:
        return winCount /float(len(race_sizes))
    
def crossVal(dat, feat, target, model, skipPct=0, n_folds=4):
    
    dat = dat.copy()
    race_ids = np.unique(dat["race_id"])
    cv = KFold(n_splits=n_folds, shuffle=True)
    scores = []
    count = 1
    for train_races, test_races in cv.split(race_ids):
        #print("CV {}/{}".format(count, n_folds))
        train_r = race_ids[train_races]
        test_r = race_ids[test_races]
        X_train = dat.loc[dat["race_id"].isin(train_r)][feat]
        y_train = dat.loc[dat["race_id"].isin(train_r)][target]
        X_test = dat.loc[dat["race_id"].isin(test_r)][feat]
        y_test = dat.loc[dat["race_id"].isin(test_r)][target]
        
        model.fit(X_train, y_train)
        
        testDat = dat.loc[dat["race_id"].isin(test_r)]
        testGroups = [len(testDat.loc[testDat["race_id"]==race_id]) for race_id in np.unique(testDat["race_id"])]
        scores.append(winnerEval(model, X_test, y_test, target, testGroups))
        count += 1
        
    meanScore = np.mean(scores)
    stdScore = np.std(scores)
    
    print("Mean score: {:.3f} +/- {:.3f}".format(meanScore, stdScore))
    
    return meanScore, stdScore


def getCrossVal(data, n_folds):
    
    from sklearn.model_selection import KFold
    
    u_races = np.unique(data["race_id"])
    cv = KFold(n_splits=n_folds, shuffle=True)
    race_splits = [(train, test) for train, test in cv.split(u_races)]
    
    # Convert race ids to run indexes
    run_splits = [(data.loc[data["race_id"].isin(u_races[train_r])],
                   data.loc[data["race_id"].isin(u_races[test_r])]) for train_r, test_r in race_splits]
    
    return run_splits

############################################# BETTING #############################################

def makeBets_1(model, X_train, y_train, testing_in, X_test, y_test, target, betSize=1, startMoney=100):
    # Fit and predict
    testing = testing_in.copy()
    betSize = float(betSize)
    model.fit(X_train, y_train[target])
    testing["predictions"] = model.predict(X_test)
    
    money = [startMoney]
    
    count = 0
    winCount = 0
    for i, r in enumerate(np.unique(testing["race_id"])):
        act = testing.loc[testing["race_id"]==r][target]
        odds = testing.loc[testing["race_id"]==r]["win_odds"]
        preds = testing.loc[testing["race_id"]==r]["predictions"]
        
        newMoney = money[-1]
        
        if newMoney < 1: # if we run out of money
            money = money + list(np.zeros(len(test_race_sizes[i:])))
            return money

        if sum(preds)==1: # One win, bet on it
            win_horse_i = np.argmax(preds)
            if act.iloc[win_horse_i]==1:
                newMoney = money[-1] - betSize + betSize*odds.iloc[win_horse_i]
                winCount += 1
            else:
                newMoney = money[-1] - betSize
                
        elif sum(preds)>1: # more than one win, find lowest odds one and bet on it
            win_horse_i = np.argmin(odds.iloc[np.where(preds==1)])
            if act.iloc[win_horse_i]==1:
                newMoney = money[-1] - betSize + betSize*odds.iloc[win_horse_i]
                winCount += 1
            else:
                newMoney = money[-1] - betSize
            
        money.append(newMoney) 
    return money

# Kellys bet
def makeBets_2(model, X_train, y_train, testing, X_test, y_test, target, startMoney=100, maxBet=5):
    # Fit and predict
    model.fit(X_train, y_train[target])
    all_preds = model.predict_proba(X_test)
    win_prob = np.array([all_preds[i][1] for i in range(len(all_preds))])
    
    money = [float(startMoney)]
    
    test_race_sizes = [len(testing.loc[testing["race_id"]==race_id]["race_id"]) for race_id in np.unique(testing["race_id"])]
    count = 0
    wCount = 0
    betCount = 0
    for s in test_race_sizes:
        lowI = count
        highI = count + s
        
        probs = win_prob[lowI:highI]
        norm_probs = [val/float(sum(probs)) for val in probs]
        
        act = y_test.iloc[lowI:highI,:][target].to_numpy()
        
        odds = testing.iloc[lowI:highI, :]["win_odds"].to_numpy()
        odds_probs = [1./val for val in odds]
        odds_probs_norm = [val/float(sum(odds_probs)) for val in odds_probs]
        
        newMoney = money[-1]
        
        for i in range(len(norm_probs)):
            
            b = odds[i] - 1
            p = norm_probs[i]
            q = 1 - p
            
            f = (b*p - q) / b
            
            if f > 0.01: # Means we should bet on this one
                betCount +=1
                bet = newMoney * f
                if bet > 1 and bet < maxBet:
                    if act[i] == 1: # Won the bet
                        wCount += 1
                        newMoney = newMoney - bet + bet*odds[i]
                    else: # Lost the bet
                        newMoney -= bet
                elif bet >=maxBet:
                    if act[i] == 1: # Won the bet
                        wCount += 1
                        newMoney = newMoney - maxBet + maxBet*odds[i]
                    else: # Lost the bet
                        newMoney -= maxBet
            
            money.append(newMoney) 
        count += s
    #print("Win Pct: {:.3f}, Loss Pct: {:.3f}".format(float(wCount)/betCount, 1-float(wCount)/betCount))
    #print("Percent bet on: {:.3f}".format(float(betCount)/len(X_test)))
    return money

# Always bet on highest prob
def makeBets_3(model, X_train, y_train, testing, X_test, y_test, target, betSize=1, startMoney=100):
    # Fit and predict
    betSize = float(betSize)
    model.fit(X_train, y_train[target])
    all_preds = model.predict_proba(X_test)
    win_prob = np.array([all_preds[i][1] for i in range(len(all_preds))])
    
    money = [startMoney]
    
    test_race_sizes = [len(testing.loc[testing["race_id"]==race_id]["race_id"]) for race_id in np.unique(testing["race_id"])]
    count = 0
    for i, s in enumerate(test_race_sizes):
        lowI = count
        highI = count + s
        preds_probs = win_prob[lowI:highI]
        winner_i = np.argmax(preds_probs)
        act = y_test.iloc[lowI:highI,:][target].to_numpy()
        odds = testing.iloc[lowI:highI, :]["win_odds"].to_numpy()
        newMoney = money[-1]
        
        if act[winner_i]==1:
            newMoney = money[-1] - betSize + betSize*odds[winner_i]
        else: 
            newMoney = money[-1] - betSize
            
        money.append(newMoney) 
        count += s
        
    return money

# If multiple bets looks at the best prob one
def makeBets_4(model, X_train, y_train, testing, X_test, y_test, target, betSize=1, startMoney=100):
    # Fit and predict
    betSize = float(betSize)
    model.fit(X_train, y_train[target])
    all_preds = model.predict(X_test)
    all_probs = model.predict_proba(X_test)
    win_prob = np.array([all_probs[i][1] for i in range(len(all_probs))])
    
    money = [startMoney]
    
    test_race_sizes = [len(testing.loc[testing["race_id"]==race_id]["race_id"]) for race_id in np.unique(testing["race_id"])]
    count = 0
    c = 0
    for i, s in enumerate(test_race_sizes):
        c += 1
        lowI = count
        highI = count + s

        preds = all_preds[lowI:highI]
        act = y_test.iloc[lowI:highI,:][target]
        odds = testing.iloc[lowI:highI, :]["win_odds"]
        newMoney = money[-1]
        
        if newMoney < 1: # if we run out of money
            #money = money + list(np.zeros(len(test_race_sizes[i:])))
            return money

        if sum(preds)==1: # One win, bet on it
            win_horse_i = np.argmax(preds)
            if act.iloc[win_horse_i]==betSize:
                newMoney = money[-1] - betSize + betSize*odds.iloc[win_horse_i]
            else:
                newMoney = money[-1] - betSize
        
        elif sum(preds)>1: # more than one win, find best prob and bet on it
            win_horse_i = np.argmax(win_prob[lowI:highI])
            if act.iloc[win_horse_i]==1:
                newMoney = money[-1] - betSize + betSize*odds.iloc[win_horse_i]
            else:
                newMoney = money[-1] - betSize
            
        money.append(newMoney) 
        count += s
        
    return money

