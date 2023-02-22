import shutil
import sys
import os
from loguru import logger

# sys.path.append("../..")
# from configs.config import DEFAULT_TABLE
DEFAULT_TABLE="milvus_obj"

def move(persons_id):
    for i in os.listdir("static/"):
        if persons_id in i:
            i = str(i)
            path = os.path.join("static",i)
            new_path = os.path.join("deleted_images",i)
            # print("move ", path, "to", new_path)
            shutil.move(path,new_path)

def do_delete(table_name, persons_id, milvus_client, mysql_cli):
    try:
       # print(table_name)
        if not table_name:
            table_name = DEFAULT_TABLE
        if not milvus_client.has_collection(table_name):
            return None
        milvus_id = mysql_cli.search_data_by_person_id(table_name, persons_id)
        # milvus_client.delete(table_name, milvus_id)
        mysql_cli.delete(table_name, milvus_id)
        move(persons_id[0])
        return "ok"
    except Exception as e:
        logger.error(" Error with delete : {}".format(e))