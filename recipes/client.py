from draft_kings_db import models, client as dk_client


def get_client(seed=False):
    client = dk_client.DraftKingsHistory(
        db_url='postgresql://user:pwd@localhost:5432/dk-learn'
    )
    if seed:
        client.initialize_nba()

    return client


def get_perf_value(perf):
    '''
    This value calculation is extremely subjective. It just tries
    to give credit to higher salary players achieving value, which
    counts for more than lower value players 
    (ex. DeMarcus Cousins at 7x value is preferable to Andre
    Roberson at 12x value).
    '''
    bonus = max(1., 1 + (perf.draft_kings_points / 8000))
    return (float(perf.draft_kings_points) / (perf.salary / 1000.)) * bonus


def get_performances(client):
    names = []
    values = []
    for res in client.session.query(models.NBAPerformance).all():
        names.append(res.name)
        values.append(res)

    return names, values


def get_tier_results(min_val, max_val, val, performances):
    total = [p for p in performances if min_val <= p.salary < max_val]
    high_value = [p for p in total if get_perf_value(p) > val]

    print(
        '{}% above {} for {} - {}'.format(
            round(float(len(high_value)) / len(total) * 100, 2),
            val,
            min_val,
            max_val
        )
    )


def print_performance_data(performances):
    print(' ')
    print('Total values: {}'.format(len(performances)))
    for v in [4, 5, 6, 7, 8]:
        print('')
        get_tier_results(3000, 4000, v, performances)
        get_tier_results(4000, 5000, v, performances)
        get_tier_results(5000, 6000, v, performances)
        get_tier_results(6000, 7000, v, performances)
        get_tier_results(7000, 8000, v, performances)
        get_tier_results(8000, 9000, v, performances)
        get_tier_results(9000, 10000, v, performances)
        get_tier_results(10000, 11000, v, performances)
        print(' ')