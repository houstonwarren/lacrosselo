from google.cloud import datastore

class Config():

    def __init__(self):
        client = datastore.Client(project='lacrosselo')
        key = client.key('Config','Config')
        vals = client.get(key)
        self.db_conn = vals['db_conn']
        self.db_pass = vals['db_pass']
        self.db_host = vals['db_host']
        self.db_name = vals['db_name']
        self.db_user = vals['db_user']
