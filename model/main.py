from os import getenv
import elo
import pymysql
from pymysql.err import OperationalError
from datetime import datetime, timedelta
import pandas as pd


CONNECTION_NAME = getenv('db_conn')
DB_HOST = getenv('db_host')
DB_USER = getenv('db_user')
DB_PASSWORD = getenv('db_pass')
DB_NAME = getenv('db_name')
SCRAPE_LINK = getenv('scrape_link')


def run_model(data,context):
    # get data and run model
    season = int(getenv('season'))
    scores, scheduled = get_games()
    rank = elo.EloRank(scores)
    rankings = rank.sorted_ranks()

    # predict upcoming games
    if len(scheduled) > 0:
        predictions = rank.predict(scheduled)
    else:
        predictions = pd.DataFrame([])

    # get performance
    performance = rank.performance(season)

    # save to database
    save_all_elo_data_to_db(rankings,predictions,performance)

    return 'done'


def get_games():
    with get_cursor() as cursor:
        # get past scores
        scores_string = 'select * from scores order by date'
        cursor.execute(scores_string)
        scores_rows = cursor.fetchall()

        # get upcoming games
        scheduled_string = '''
            select * 
            from scheduled 
            where date >= curdate()
                and date < date_add(curdate(), interval 7 day)
        '''
        cursor.execute(scheduled_string)
        scheduled_rows = cursor.fetchall()

        # return as dataframes
        scores = pd.DataFrame(scores_rows)
        scores['date'] = pd.to_datetime(scores['date'])
        scheduled = pd.DataFrame(scheduled_rows)
        if len(scheduled) > 0:
            scheduled['date'] = pd.to_datetime(scheduled['date'])
        return scores,scheduled


def save_all_elo_data_to_db(ranks,predictions,performance):
    season = int(getenv('season'))
    with get_cursor() as cursor:

        # update rankings
        for i, rank in ranks.iterrows():
            team = rank['team']
            rank = rank['rank']
            # sanitize quotes in string
            if "'" in team:
                team = sanitize_quoted_string(team)

            rank_update_str = f'''
                update ranks
                set rank = {rank}
                where team = '{team}'
            '''
            cursor.execute(rank_update_str)

        # update predictions
        for i, prediction in predictions.iterrows():
            team1 = prediction['team1']
            team2 = prediction['team2']
            # sanitize quotes in string
            if "'" in team1:
                team1 = sanitize_quoted_string(team1)

            if "'" in team2:
                team2 = sanitize_quoted_string(team2)

            prediction_update_string = f'''
                INSERT IGNORE
                INTO predictions (date,team1,team2,team1_rank,team2_rank,win_prop,line) 
                VALUES (
                    '{prediction['date']}',
                    '{team1}',
                    '{team2}',
                    {prediction['team1_rank']},
                    {prediction['team2_rank']},
                    {prediction['win_prop']},
                    {prediction['line']}
                );
            '''

            cursor.execute(prediction_update_string)

        # update brackets
        bins = performance['bins']
        for bin in bins:
            bin_update_str = f'''
                update brackets
                set
                    num_games = {bin[2]},
                    predicted = {bin[3]},
                    pct = {bin[4]},
                    games_off = {bin[5]}
                where
                    bin_start = {bin[0]} and bin_end = {bin[1]}
            '''
            cursor.execute(bin_update_str)

        # update performance
        performance_update_str = f'''
            update performance
            set
                brier = {performance['brier']},
                brier_decomp = {performance['brier_decomp']},
                games_correct = {performance['games_correct']},
                games_incorrect = {performance['games_incorrect']},
                accuracy = {performance['accuracy']},
                games_off = {performance['games_off']},
                games_off_pct = {performance['games_off_pct']}
            where
                season = {season}
        '''
        cursor.execute(performance_update_str)


def sanitize_quoted_string(quote_string):
    loc = quote_string.index("'")
    team = quote_string[:loc] + "'" + quote_string[loc:]
    return team


def get_cursor():
    """
    cursor function for database access. Taken from cloud function documentation

    :return: pymysql.connect.cursor
    """

    mysql_config = {
        'host': DB_HOST,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'db': DB_NAME,
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor,
        'autocommit': True,
        'unix_socket': f'/cloudsql/{CONNECTION_NAME}'
    }

    try:
        conn = pymysql.connect(**mysql_config)
        return conn.cursor()
    except OperationalError:
        conn = pymysql.connect(**mysql_config)
        conn.ping(reconnect=True)
        return conn.cursor()
