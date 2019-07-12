import pymysql
from pymysql.err import OperationalError
from os import getenv
from gc_config import Config

# get environment variables and establish connection vars
MODE = getenv('MODE')
config = Config()
CONNECTION_NAME = config.db_conn
DB_HOST = config.db_host
DB_USER = config.db_user
DB_PASSWORD = config.db_pass
DB_NAME = config.db_name

# create sql config depending on environment
if MODE == 'PROD':
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

elif MODE=='DEV':
    mysql_config = {
        'host': DB_HOST,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'db': DB_NAME,
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor,
        'autocommit': True
    }

else:
    raise Exception('MODE environment variable not valid')


def get_rankings():
    # get the data
    sql_string = 'select * from ranks order by rank desc'
    data = sql_to_dict_list(sql_string)

    return data


def get_predictions():
    sql_string = '''
        select * 
        from predictions 
        where date >= curdate()
            and date < date_add(curdate(), interval 7 day)
    '''
    data = sql_to_dict_list(sql_string)

    return data


def get_performance():
    # get the calibration bracket info
    bracket_sql_string = 'select * from brackets order by bin_start asc'
    bracket_data = sql_to_dict_list(bracket_sql_string)

    # get the general accuracy data
    accuracy_sql_string = 'select * from performance'
    accuracy_data = sql_to_dict_list(accuracy_sql_string)

    return bracket_data, accuracy_data


def sql_to_dict_list(sql_string):
    """

    :param sql_output: result of a cursor.fetchall() on a query
    :return: a list of dicts where each dict represents a row in the data
    """

    # initiate the cursor object
    with get_cursor() as cursor:
        cursor.execute(sql_string)
        rows = cursor.fetchall()
        data_dict_list = []

        # make sure there is data to be retrieved, otherwise return an empty list
        if len(rows) > 0:
            cols = [z[0] for z in cursor.description]

            for row in rows:
                data_dict = {}
                for i, val in enumerate(row.values()):
                    data_dict[cols[i]] = val
                data_dict_list.append(data_dict)

    return data_dict_list


def get_cursor():
    """
    cursor function for database access. Taken from cloud function documentation

    :return: pymysql.connect.cursor
    """

    try:
        conn = pymysql.connect(**mysql_config)
        return conn.cursor()
    except OperationalError:
        conn = pymysql.connect(**mysql_config)
        conn.ping(reconnect=True)
        return conn.cursor()


