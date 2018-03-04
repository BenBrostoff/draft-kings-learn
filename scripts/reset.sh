#!/usr/bin/env bash

docker-compose down
docker-compose up -d --build

python - << EOF
# generate pickles
from recipes import linear_regressor
linear_regressor.run()
EOF
