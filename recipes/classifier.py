from sklearn import tree
import draft_kings_db

# TODO - Dockerize this setup to avoid requests from S3
client = draft_kings_db.client.DraftKingsHistory()
client.initialize_nba()

features = []
labels = []


def get_perf_value(perf):
    return float(perf.draft_kings_points) / (perf.salary / 1000)


names = ['Kevin Durant', 'Anthony Davis', 'Marcus Smart', 'Tyreke Evans']

for name in names:
    performances = client.lookup_nba_performances(name)
    last = performances[0]
    prev = performances[1]
    features.append([
        # given tonight's salary
        last.salary,
        # try to predict on last game's points, minutes, salary
        prev.salary,
        prev.draft_kings_points,
        prev.minutes,
    ])
    labels.append(0 if get_perf_value(last) < 5 else 1)

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

# try some examples
print(
    clf.predict([[8000, 7900, 50, 35]])
)
from IPython import embed; embed();