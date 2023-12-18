import pandas as pd
import pymongo
import json
import os

from dataclasses import dataclass

class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")
    #mongo_db_url = 'mongodb://localhost:27017'
    

env_var = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
print(env_var.mongo_db_url)

TARGET_COLUMN = "expenses"