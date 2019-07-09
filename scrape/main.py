from os import getenv
import scrape
import pymysql
from pymysql.err import OperationalError
from google.cloud import pubsub_v1


CONNECTION_NAME = getenv('db_conn')
DB_HOST = getenv('db_host')
DB_USER = getenv('db_user')
DB_PASSWORD = getenv('db_pass')
DB_NAME = getenv('db_name')
SCRAPE_LINK = getenv('scrape_link')


def scrape_upload(data,context):
    """
    Invocation function for cloud function to scrape games and upcoming, and write to database

    :return: None
    """

    # scrape + parse and do some housekeeping around names, d1 teams
    games, scheduled = scrape.scrape_link(SCRAPE_LINK)
    games, scheduled = scrape.harmonize_team_names(games, scheduled)
    games, scheduled = scrape.delete_if_not_d1(games, scheduled, scrape.d1)

    # upload to db
    upload(games,scheduled)

    # invoke model run using gcloud pubsub
    push_to_pubsub()

    return "done"


def upload(games,scheduled):
    """
    Structure sql_strings from dataframes and insert into database. Skip duplicate values

    :param games: played games as a pd.DataFrame
    :param scheduled: upcoming scheduled games as a pd.DataFrame
    :return: None
    """
    # write played games not already in the database to db
    for i,played in games.iterrows():
        with get_cursor() as cursor:
            sql_string = f'''
                INSERT IGNORE
                INTO scores (date, team1, team1_score,team2,team2_score,ot,number_ot,season,neutral,tourney) 
                VALUES (
                    '{played['date']}',
                    '{played['team1']}',
                    {played['team1_score']},
                    '{played['team2']}',
                    {played['team2_score']},
                    {played['ot']},
                    {played['number_ot']},
                    {played['season']},
                    {played['neutral']},
                    {played['tourney']}
                );
            '''
            cursor.execute(sql_string)

    # write scheduled games not already in the database to db
    for i,upcoming in scheduled.iterrows():
        with get_cursor() as cursor:
            sql_string = f'''
                INSERT IGNORE
                INTO scheduled (date, team1,team2,season,neutral,tourney) 
                VALUES (
                    '{upcoming['date']}',
                    '{upcoming['team1']}',
                    '{upcoming['team2']}',
                    {upcoming['season']},
                    {upcoming['neutral']},
                    {upcoming['tourney']}
                );
            '''
            cursor.execute(sql_string)


def push_to_pubsub():
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path('lacrosselo', 'runmodel-pubsub')
    data = '{}'.encode('utf-8')
    publisher.publish(topic_path, data=data)


def get_cursor():
    """
    cursor function for database access. Taken from cloud function documentation

    :return:
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