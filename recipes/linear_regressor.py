import pandas as pd
from sklearn import datasets ## imports datasets from scikit-learn
from sklearn import linear_model
from client import (
    get_client,
    get_performances,
    get_perf_value,
    print_performance_data,
)
import numpy as np

client = get_client(seed=False)
names, performances = get_performances(client)

data = datasets.load_boston() ## loads Boston dataset from datasets library
# print(data.data) /

# define the data/predictors as the pre-set feature names
LOOK_BACK = 3
use = []
target = []
for name in set(names):
    performances = client.lookup_nba_performances(name, limit=None)
    total = len(performances)

    for idx in range(total):
        if idx < total - LOOK_BACK:
            last = performances[idx]
            prev = performances[idx + 1]
            prev_three = performances[(idx + 1):(idx + LOOK_BACK + 1)]

            if last.draft_kings_points == 0 or (
                0 in [p.draft_kings_points for p in prev_three]
            ):
                continue
            target.append(last.draft_kings_points)
            use.append([
                # last.draft_kings_points,
                last.salary,
                prev.draft_kings_points,
                # prev.salary,
                # prev.draft_kings_points,
                # prev.minutes,
                # prev.rebounds,
                # prev.assists,
                # prev.blocks,
                # prev.steals,
                # prev.turnovers,

                # numpy.mean([x.salary for x in prev_three]),
                # numpy.std([x.draft_kings_points for x in prev_three]),
                # numpy.mean([x.draft_kings_points for x in prev_three]),
                # numpy.mean([x.minutes for x in prev_three]),
                # numpy.mean([x.rebounds for x in prev_three]),
                # numpy.mean([x.assists for x in prev_three]),
                # numpy.mean([x.blocks for x in prev_three]),
                # numpy.mean([x.steals for x in prev_three]),
                # numpy.mean([x.turnovers for x in prev_three]),
            ])

            # val = get_perf_value(last)
            # if val < 4.5:
            #     labels.append(0)
            # else:
            #     labels.append(1)

df = pd.DataFrame(use, columns=['LAST_SAL', 'PREV_P'])
print(df)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(target, columns=["LAST_P"])

X = df
y = target['LAST_P']

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

print(lm.coef_)
print(lm.predict([[9000, 50]]))