from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta

db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'database': 'rec_sys',
}

engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

def run_query(query):
    return pd.read_sql_query(query, con=engine)
