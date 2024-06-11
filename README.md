## Predicting NBA MVP

Predicting the MVP of a season using Machine Learning (Sklearn) with data from 2019-20 season to 2023-24 season for all players and data from the 1955-56 season for MVP's

[source](https://stathead.com/tiny/bzJUn) [source 2](https://stathead.com/tiny/6Jswm) *(Combined into the same dataframe)* 

### Glossary
- Player -- Name of Player
- Season -- Season Stats are from
- Age -- Player's age on February 1 of the season
- Team -- Team(s) played in during the season
- G -- Games
- GS -- Games Started
- AS -- All-Star Team Selections
- MP -- Minutes Played
- FG -- Field Goals
- FGA -- Field Goal Attempts
- 2P -- 2-Point Field Goals
- 2PA -- 2-Point Field Goal Attempts
- 3P -- 3-Point Field Goals
- 3PA -- 3-Point Field Goal Attempts
- FT -- Free Throws
- FTA -- Free Throw Attempts
- ORB -- Offensive Rebounds
- DRB -- Defensive Rebounds
- TRB -- Total Rebounds
- AST -- Assists
- STL -- Steals
- BLK -- Blocks
- TOV -- Turnovers
- PF -- Personal Fouls
- PTS -- Points
- FG% -- Field Goal Percentage
- 2P% -- 2-Point Field Goal Percentage
- 3P% -- 3-Point Field Goal Percentage
- FT% -- Free Throw Percentage
- TS% -- True Shooting Percentage
  - A measure of shooting efficiency that takes into account 2-point field goals, 3-point field goals, and free throws.
- eFG% -- Effective Field Goal Percentage
  - This statistic adjusts for the fact that a 3-point field goal is worth one more point than a 2-point field goal.


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

There is missing data, especially from older seasons that I tried to avoid adding to the model, however due to the amount of missing data that could not be entirely avoided, therefore the void data was filled in with 0's. A list of how much data is missing from each column is below.

The data is filtered to only include players seasons where they played in over 70 games. This is in order to filter those who haven't played enough games in order to be a viable contender for MVP. Season is also changed into a single number, the year the season ends. (ex. 2023-24 is 24)


```python
df = pd.read_csv('data.csv')
del df['Rk']
na = df.isna().sum()
df.fillna(0, inplace=True)
df = df[df['G'] > 70]
def season(season):
    return season[-2::]
df['Season'] = df['Season'].apply(season)
df = df.reset_index(drop=True)
na
```




    Player      0
    Season      0
    Age         0
    Team        0
    G           0
    GS         25
    AS          0
    MP          0
    FG          0
    FGA         0
    2P          0
    2PA         0
    3P         24
    3PA        24
    FT          0
    FTA         0
    ORB        18
    DRB        18
    TRB         0
    AST         0
    STL        18
    BLK        18
    TOV        22
    PF          0
    PTS         0
    FG%        18
    2P%        34
    3P%       155
    FT%       154
    TS%        17
    eFG%       18
    Pos         0
    MVP         0
    dtype: int64



The model is a Pipeline first applying a standard Scalar then a Linear Regression.

A standard scaler centers the data around 0 then makes the variance of the data 1, allowing the machine to easily understand the data is comparison to other columns. The following transformation is applied:

$$X_{scaled} = \frac{X - \mu }{ \sigma} $$

After the data is made uniform, a linear regression is applied, this assumes there is a relationship between the data and finds the relationship using ordinary least squares linear regression. Sklearn forms the following formula:

$$ y = Xw+b $$

Where \
$y = $ the dependant variable (Whether or not a player is MVP) \
$X = $ the matrix of input features \
$w = $ the vector of coeffiecents \
$b = $ the intercept term

This formula can then be used on new data to predict the dependant variable.


```python
model = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])

model.fit(df.loc[:,['eFG%', 'TS%','PTS','2P','AS','TRB','Age','Season','AST','MP','TOV','PF','G','FT','FTA','3PA']],pd.DataFrame(df['MVP']))
```




<style>#sk-container-id-32 {color: black;}#sk-container-id-32 pre{padding: 0;}#sk-container-id-32 div.sk-toggleable {background-color: white;}#sk-container-id-32 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-32 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-32 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-32 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-32 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-32 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-32 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-32 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-32 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-32 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-32 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-32 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-32 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-32 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-32 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-32 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-32 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-32 div.sk-item {position: relative;z-index: 1;}#sk-container-id-32 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-32 div.sk-item::before, #sk-container-id-32 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-32 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-32 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-32 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-32 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-32 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-32 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-32 div.sk-label-container {text-align: center;}#sk-container-id-32 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-32 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-32" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;scaler&#x27;, StandardScaler()),
                (&#x27;regression&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-94" type="checkbox" ><label for="sk-estimator-id-94" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;scaler&#x27;, StandardScaler()),
                (&#x27;regression&#x27;, LinearRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-95" type="checkbox" ><label for="sk-estimator-id-95" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-96" type="checkbox" ><label for="sk-estimator-id-96" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div></div></div>




```python
score = model.score(df.loc[:,['eFG%', 'TS%','PTS','2P','AS','TRB','Age','Season','AST','MP','TOV','PF','G','FT','FTA','3PA']],df['MVP'])
str(round(score*100,2)) + '%'
```




    '75.93%'



The model is 76% correct at predicting past MVP's, and can be used to get an idea on who will be MVP for a season. It can't be perfect because MVP is the opinion of a group of people therefore the model is trying to predict human decisions based on opinion without any information about the people.
