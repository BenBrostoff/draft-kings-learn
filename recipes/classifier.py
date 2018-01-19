from sklearn import tree
import draft_kings_db
from draft_kings_db import models

# TODO - Dockerize this setup to avoid requests from S3
client = draft_kings_db.client.DraftKingsHistory()
client.initialize_nba()

features = []
labels = []


def get_perf_value(perf):
    return float(perf.draft_kings_points) / (perf.salary / 1000.)

names = []
for res in client.session.query(models.NBAPerformance).all():
    names.append(res.name)

for name in set(names):
    performances = client.lookup_nba_performances(name, limit=10)
    if len(performances) < 3:
        continue
    last = performances[0]
    prev = performances[1]
    second_prev = performances[2]

    features.append([
        # given tonight's salary
        last.salary,
        # try to predict on last game's points, minutes, salary
        prev.salary,
        prev.draft_kings_points,
    ])
    labels.append(0 if get_perf_value(last) < 5 else 1)


    features.append([
        # given tonight's salary
        prev.salary,
        # try to predict on last game's points, minutes, salary
        second_prev.salary,
        second_prev.draft_kings_points,
    ])
    labels.append(0 if get_perf_value(prev) < 5.5 else 1)

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

# try some examples
for sal in [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]:
    for last in [30, 40, 50, 60]:
        print(
            '{} {}: {}'.format(
                sal,
                last,
                clf.predict([[sal, sal, 40]])
            )
        )