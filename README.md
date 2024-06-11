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
$y =$ the dependant variable (Whether or not a player is MVP) \
$X =$ the matrix of input features \
$w =$ the vector of coeffiecents \
$b =$ the intercept term

This formula can then be used on new data to predict the dependant variable.


```python
model = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])

model.fit(df.loc[:,['eFG%', 'TS%','PTS','2P','AS','TRB','Age','Season','AST','MP','TOV','PF','G','FT','FTA','3PA']],pd.DataFrame(df['MVP']))
```


```python
score = model.score(df.loc[:,['eFG%', 'TS%','PTS','2P','AS','TRB','Age','Season','AST','MP','TOV','PF','G','FT','FTA','3PA']],df['MVP'])
str(round(score*100,2)) + '%'
```




    '75.93%'



The model is 76% correct at predicting past MVP's, and can be used to get an idea on who will be MVP for a season. It can't be perfect because MVP is the opinion of a group of people therefore the model is trying to predict human decisions based on opinion without any information about the people.
