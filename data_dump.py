import pymongo
import pandas as pd
import json
from india_insurance.config import mongo_client

DATABASE_NAME = "india_insurance"
COLLECTION_NAME = "premium"
DATA_FILE_PATH = "/notebook/insurance-premium-prediction/insurance.csv"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns : {df.shape}")

    df.reset_index(drop=True,inplace=True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

    