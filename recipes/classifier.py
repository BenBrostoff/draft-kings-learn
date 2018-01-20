import pickle
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
values = []
for res in client.session.query(models.NBAPerformance).all():
    names.append(res.name)
    values.append(res)


def get_tier_results(min_val, max_val, val):
    total = [p for p in values if p.salary >= min_val and p.salary < max_val]
    high_value = [p for p in total if get_perf_value(p) > val]

    print(
        '{}% above {} for {} - {}'.format(
            round(float(len(high_value)) / len(total), 2) * 100,
            val,
            min_val,
            max_val
        )
    )

print(' ')
print('Total values: {}'.format(len(values)))
get_tier_results(3000, 4000, 5)
get_tier_results(4000, 5000, 5)
get_tier_results(5000, 6000, 5)
get_tier_results(6000, 7000, 5)
get_tier_results(7000, 8000, 5)
get_tier_results(8000, 9000, 5)
get_tier_results(9000, 10000, 5)
get_tier_results(10000, 11000, 5)
print(' ')

LOOK_BACK = 12
for name in set(names):
    performances = client.lookup_nba_performances(name, limit=LOOK_BACK)
    if len(performances) < LOOK_BACK:
        continue

    for idx in range(LOOK_BACK):
        if idx < LOOK_BACK - 2:
            last = performances[idx]
            features.append([
                performances[idx].salary,
                performances[idx + 1].salary,
                performances[idx + 1].draft_kings_points,
            ])
            labels.append(0 if get_perf_value(last) < 5 else 1)

print('Total performances: {}'.format(len(labels)))


clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

# try some examples
for sal in [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]:
    for last in [30, 40, 50, 60]:
        print(
            '{} {}: {}'.format(
                sal,
                last,
                clf.predict([[sal, sal, last]])
            )
        )
    print(' ')


# save as pickle for use in other projects
pickle.dump({'clf': clf}, open('clf.pickle', 'wb'))
