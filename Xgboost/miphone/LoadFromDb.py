from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np
db_connection_str = 'mysql+pymysql://root:12345678@localhost/miphone'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM brand_price', con=db_connection)

# print(df)
# print(np.array(df.iloc[:,-1]))
print(np.array(df.iloc[:,:-1]))
