import numpy as np
import pandas as pd
import math as m

from sklearn.model_selection import KFold
############## CROSS VALIDATION FUNCTION ##############
def getCrossVal(data, n_folds):
    
    u_races = np.unique(data["race_id"])
    cv = KFold(n_splits=n_folds, shuffle=True)
    race_splits = [(train, test) for train, test in cv.split(u_races)]
    
    # Convert race ids to run indexes
    run_splits = [(data.loc[data["race_id"].isin(u_races[train_r])],
                   data.loc[data["race_id"].isin(u_races[test_r])]) for train_r, test_r in race_splits]
    
    return run_splits

############## EVALUATE MODEL ACCURACY AS GUESSING BEST HORSE ##############
def winnerEval(winPreds, y_test, data):
    # convert preds into an actual win choice
    race_sizes = [data.loc[data["race_id"]==r].shape[0] for r in np.unique(data["race_id"])]
    winCount = 0
    temp = 0 
    for i, s in enumerate(race_sizes):
        low_index = temp
        high_index = temp + s
        
        racePreds = winPreds[low_index:high_index]
        raceVals = y_test[low_index:high_index]
        
        predWinner = np.argmax(racePreds, axis=0)
        actWinner = np.argmax(raceVals, axis=0)

        if predWinner == actWinner:
            winCount += 1
                    
        temp += s
        
    return winCount/float(len(race_sizes))

############## R SQUARED CALCULATION ##############
def RSq(preds, data, a=1, b=1):
    data = data.copy().reset_index(drop=True)
    preds = preds.copy().reset_index(drop=True)
    
    # Get random preds
    randPreds = getRandPreds(data)
    
    LLmodel = L(preds, data, a, b)
    LLbl = L(randPreds, data, a, b)
    
    return 1 - LLmodel/LLbl

############## LOG LIKLIHOOD FUNCTION ##############
def L(preds, data, a=1, b=1):
    # Calculate the LL 
    data = data.copy().reset_index(drop=True)
    preds = preds.copy().reset_index(drop=True)
    races = np.unique(data["race_id"])
    
    LLmodel = 0.
    for r in races:
        
        race = data.loc[data["race_id"]==r] 
        raceI = race.index
        raceSize = race.shape[0]
        
        winnerIndex = race.loc[race["won"]==1].index
        if winnerIndex.shape[0]!=1:
            continue
            
        modelPreds = preds.iloc[winnerIndex].values
        
        # Get the relevant coefficents
        modelSum = 0.
        
        odds = race["win_odds"]
        probs = [1./o for o in odds]
        normProbs = [p/sum(probs) for p in probs]
        for i in range(raceSize):
            # Get odds probs
            index = raceI[i]
            oddsProbability = normProbs[i]
            if index==winnerIndex:
                winningProb = oddsProbability
                
            modelSum += m.exp(a*m.log(preds[index]) + b*m.log(oddsProbability))
            
        # Now get the numerator
        modelNumer = m.exp(a*m.log(modelPreds) + b*m.log(winningProb))
        c = modelNumer/modelSum
        
        LLmodel += m.log(c)
        
    return LLmodel

############## GET IMPLIED ODDS PROBABILITY ##############
def oddsProbs(testing):
    testing = testing.copy()
    races = np.unique(testing["race_id"])
    testing["probs"] = 0.
    for r in races:
        race = testing.loc[testing["race_id"]==r]
        raceI = race.index
        raceSize = race.shape[0]
        odds = race["win_odds"]
        oddsProbs = [1./i for i in odds.values]
        normOddsProbs = [i/sum(oddsProbs) for i in oddsProbs]
        
        for i in range(raceSize):
            ind = raceI[i]
            p = normOddsProbs[i]
            testing.loc[ind, "probs"] = p
    
    return testing["probs"]
    
############## RANDOM GUESSING PROBABILITY ##############
def getRandPreds(data):
    data = data.copy().reset_index(drop=True)
    
    # Get LL of random preds
    randomProbs = np.ones(len(data))
    races = np.unique(data["race_id"])
    for r in races:
        race = data.loc[data["race_id"]==r]
        raceI = race.index
        raceSize = race.shape[0]
        if raceSize!=0:
            randomPreds = 1./raceSize
            randomProbs[raceI] = randomPreds
            
    return pd.Series(randomProbs)


#################################### COMBINE PROBS ####################################
def combineProbs(preds, data, a=1, b=1):
    data = data.copy().reset_index(drop=True)
    preds = preds.copy().reset_index(drop=True)
    odds = oddsProbs(data)
    races = np.unique(data["race_id"])
    output = np.zeros(len(preds))
    
    for r in races:
        race = data.loc[data["race_id"]==r]
        raceSize = race.shape[0]
        raceI = race.index
        
        #modelPreds = preds.loc[preds.index==raceI]
        modelPreds = preds[raceI].to_numpy()
        oddsPreds = odds[raceI].to_numpy()
        
        s = 0.
        for i in range(raceSize):
            mPred = modelPreds[i]
            oPred = oddsPreds[i]
            
            s += m.exp(a*m.log(mPred) + b*m.log(oPred))
            
        combinePreds = [m.exp(a*m.log(modelPreds[i]) + b*m.log(oddsPreds[i]))/s for i in range(raceSize)]
        
        for i in range(raceSize):
            ix = raceI[i]
            output[ix] = combinePreds[i]
            
    return pd.Series(output)

#################################### BETTING SIMULATION ####################################
# KELLYS BET WITH RESTRICTIONS
def betSim_1(preds, data, minBetSize=0.5, maxBetSize=5, startMoney=100):
    data = data.copy().reset_index(drop=True)
    races = np.unique(data["race_id"])
    money = [float(startMoney)]
    betsPlaced = 0.
    betsWon = 0.
    for r in races:
        race = data.loc[data["race_id"]==r]
        raceSize = race.shape[0]
        raceI = race.index
    
        modelPreds = np.array([i for _, i in preds[raceI]])
        modelPreds = [i/sum(modelPreds) for i in modelPreds]
        
        actOdds = data.iloc[raceI]["win_odds"].to_numpy()
        results = data.iloc[raceI]["won"].to_numpy()
        if sum(results)!=1:
            continue
        
        winnerI = np.argmax(results)
        
        for i in range(raceSize):
            myProb = modelPreds[i]
            decOdds = actOdds[i]
            
            advantage = myProb*decOdds - 1. 
            K = advantage / (decOdds - 1.)
            if K <= 0:
                continue
            recBet = money[-1] * K
            
            if recBet > minBetSize: # Means we are gonna bet
                betsPlaced += 1
                didWin = (winnerI == i)
                #recBet *= 0.4
                
                if recBet < maxBetSize:
                    # Place rec bet
                    actualBet = recBet
                else:
                    # Place max bet
                    actualBet = maxBetSize
                    
                if didWin:
                    betsWon += 1
                    newMoney = money[-1] + actualBet * (decOdds - 1)
                else:
                    newMoney = money[-1] - actualBet
                    
                money.append(newMoney)
                
    if betsPlaced>2:
        print("Win Percent: {}".format(betsWon/betsPlaced))
    return money # Thats the goal ...
    
# KELLYS BET RESTRICTED AND ONLY BET ONCE PER RACE
def betSim_2(preds, data, minBetSize=0.5, maxBetSize=5, startMoney=100, betModifier=1., minProb=0.):

    race_sizes = [data.loc[data["race_id"]==r].shape[0] for r in np.unique(data["race_id"])]
    money = [float(startMoney)]
    
    s, temp = 0, 0
    betsPlaced = 0.
    secondBetCounter = 0
    betsWon = 0.
    for i, s in enumerate(race_sizes):
        low_index = temp
        high_index = temp + s
        temp += s
        
        racePreds = preds[low_index:high_index].to_numpy()
        raceVals = data.iloc[low_index:high_index, :]["won"].to_numpy()
        raceOdds = data.iloc[low_index:high_index, :]["win_odds"].to_numpy()
 
        predWinner = np.argmax(racePreds, axis=0)
        actWinner = np.argmax(raceVals, axis=0)
        if predWinner == actWinner:
            secondBetCounter += 1
        
        predOdds = raceOdds[predWinner]
        predProb = racePreds[predWinner]

        if sum(raceVals)!=1:
            continue
        B = predOdds - 1.
        P = predProb
        Q = 1. - predProb
        K = (B*P - Q) / B
        
        if K > 0 and money[-1]*K > minBetSize and P > minProb:  # Means we are gonna bet
            recBet = money[-1] * K # reccomended bet
            
            betsPlaced += 1
            didWin = (predWinner == actWinner)
            recBet *= betModifier

            if recBet < maxBetSize:
                # Place rec bet
                actualBet = recBet
            else:
                # Place max bet
                actualBet = maxBetSize

            if didWin:
                betsWon += 1
                newMoney = money[-1] + actualBet * (predOdds - 1)
            else:
                newMoney = money[-1] - actualBet

            money.append(newMoney)
                
    if betsPlaced>2:
        print("Win Percent: {}".format(betsWon/betsPlaced))
        print(secondBetCounter/float(len(race_sizes)))
    return money # Thats the goal ...
    
# BET A SET AMMOUNT ON THE BEST RACE OPTION
def betSim_3(preds, data, betSize=1, startMoney=100):
    preds = [i for _, i in preds]

    race_sizes = [data.loc[data["race_id"]==r].shape[0] for r in np.unique(data["race_id"])]
    money = [float(startMoney)]
    
    s, temp = 0, 0
    betsWon = 0.
    for i, s in enumerate(race_sizes):
        low_index = temp
        high_index = temp + s
        temp += s
        
        racePreds = preds[low_index:high_index]
        raceVals = data.iloc[low_index:high_index, :]["won"].to_numpy()
        raceOdds = data.iloc[low_index:high_index, :]["win_odds"].to_numpy()
 
        predWinner = np.argmax(racePreds, axis=0)
        actWinner = np.argmax(raceVals, axis=0)
        predOdds = raceOdds[predWinner]
        predProb = racePreds[predWinner]
        if predWinner == actWinner:
            betsWon += 1
            newMoney = money[-1] + (predOdds-1)*betSize
        else:
            newMoney = money[-1] - betSize
            
        money.append(newMoney)
                
    print("Win Percent: {}".format(betsWon/float(len(race_sizes))))
    return money # Thats the goal ...



