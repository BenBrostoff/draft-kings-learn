import sys
import pickle
from collections import Counter
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy
from recipes.client import (
    get_client,
    get_performances,
    get_perf_value,
    print_performance_data,
)


def run():
    seed = False
    if len(sys.argv) > 1 and sys.argv[1] == 'seed':
        seed = True

    client = get_client(seed=seed)

    features = []
    labels = []

    names, performances = get_performances(client)
    # print_performance_data(performances)

    LOOK_BACK = 3
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

                features.append([
                    last.salary,
                    prev.salary,
                    prev.draft_kings_points,
                    prev.minutes,
                    prev.rebounds,
                    prev.assists,
                    prev.blocks,
                    prev.steals,
                    prev.turnovers,

                    numpy.mean([x.salary for x in prev_three]),
                    numpy.std([x.draft_kings_points for x in prev_three]),
                    numpy.mean([x.draft_kings_points for x in prev_three]),
                    numpy.mean([x.minutes for x in prev_three]),
                    numpy.mean([x.rebounds for x in prev_three]),
                    numpy.mean([x.assists for x in prev_three]),
                    numpy.mean([x.blocks for x in prev_three]),
                    numpy.mean([x.steals for x in prev_three]),
                    numpy.mean([x.turnovers for x in prev_three]),
                ])

                val = get_perf_value(last)
                if val < 4.5:
                    labels.append(0)
                else:
                    labels.append(1)

    print(
        Counter(labels)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
    )

    print('Total performances: {}'.format(len(labels)))


    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(
        'Score: {}'.format(clf.score(X_test, y_test)),
    )

    # save as pickle for use in other projects
    pickle.dump({'clf': clf}, open('clf.pickle', 'wb'))
