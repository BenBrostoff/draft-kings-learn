import pandas as pd
from sklearn import linear_model
import numpy
from .client import (
    get_client,
    get_performances,
)


PREDICTOR_COLUMNS = [
    'LAST_MINUTES',
    'LAST_POINTS',
    'LAST_REBOUNDS',
    'LAST_ASSISTS',
    'LAST_BLOCKS',
    'LAST_STEALS',
    'LAST_TURNOVERS',
    'LAST_THREE_MINUTES',
    'LAST_THREE_POINTS',
    'LAST_THREE_REBOUNDS',
    'LAST_THREE_ASSISTS',
    'LAST_THREE_BLOCKS',
    'LAST_THREE_STEALS',
    'LAST_THREE_TURNOVERS',
]

TARGET_COLUMN = ['POINTS']


def run(seed=True, look_back=3):
    client = get_client(seed)  # TODO - make optional in util
    names, performances = get_performances(client)

    use = []
    target = []
    for name in set(names):
        performances = client.lookup_nba_performances(name, limit=None)
        total = len(performances)

        for idx in range(total):
            if idx < total - look_back:
                last = performances[idx]
                prev = performances[idx + 1]
                prev_three = performances[(idx + 1):(idx + look_back + 1)]

                if last.draft_kings_points == 0 or (
                    0 in [p.draft_kings_points for p in prev_three]
                ):
                    continue
                target.append(last.draft_kings_points)
                use.append([
                    prev.minutes,
                    prev.points,
                    prev.rebounds,
                    prev.assists,
                    prev.blocks,
                    prev.steals,
                    prev.turnovers,
                    numpy.mean([x.minutes for x in prev_three]),
                    numpy.mean([x.points for x in prev_three]),
                    numpy.mean([x.rebounds for x in prev_three]),
                    numpy.mean([x.assists for x in prev_three]),
                    numpy.mean([x.blocks for x in prev_three]),
                    numpy.mean([x.steals for x in prev_three]),
                    numpy.mean([x.turnovers for x in prev_three]),
                ])

    df = pd.DataFrame(use, columns=PREDICTOR_COLUMNS)
    target = pd.DataFrame(target, columns=TARGET_COLUMN)

    # TODO - split into test, training
    X = df
    y = target['POINTS']

    lm = linear_model.LinearRegression()
    model = lm.fit(X, y)

    return model


def print_coeff(model):
    for idx, c in enumerate(model.coef_):
        print('{}: {}'.format(PREDICTOR_COLUMNS[idx], c))
