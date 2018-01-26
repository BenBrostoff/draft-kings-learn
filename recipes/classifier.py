import sys
import pickle
from collections import Counter
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import numpy
import pydotplus
from IPython.display import Image
from client import (
    get_client,
    get_performances,
    get_perf_value,
    print_performance_data,
)

seed = False
if len(sys.argv) > 1 and sys.argv[1] == 'seed':
    seed = True

client = get_client(seed=seed)

features = []
labels = []

names, performances = get_performances(client)
print_performance_data(performances)

for name in set(names):
    performances = client.lookup_nba_performances(name, limit=None)
    total = len(performances)
    for idx in range(total):
        if idx < total - 2:
            last = performances[0]
            if last.draft_kings_points == 0:
                continue
            prev_three = performances[1:4]
            val = get_perf_value(last)
            features.append([
                performances[idx].salary,
                performances[1].salary,
                performances[1].draft_kings_points,
                performances[1].minutes,
                performances[1].rebounds,
                performances[1].assists,
                performances[1].blocks,
                performances[1].steals,
                performances[1].turnovers,

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
            if val < 4:
                labels.append(0)
            elif val < 5:
                labels.append(1)
            elif val < 6:
                labels.append(2)
            else:
                labels.append(3)

print(
    Counter(labels)
)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.05,
    random_state=42
)


print('Total performances: {}'.format(len(labels)))


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(
    'Score: {}'.format(clf.score(X_test, y_test)),
)

# # save as pickle for use in other projects
# pickle.dump({'clf': clf}, open('clf.pickle', 'wb'))
#
# # visualize
# dot_data = StringIO()
# export_graphviz(
#     clf,
#     out_file=dot_data,
#     filled=True,
#     rounded=True,
#     special_characters=True
#     )
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())