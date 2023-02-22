import sys

from loguru import logger

# sys.path.append("../..")
# from configs.config import DEFAULT_TABLE
DEFAULT_TABLE="milvus_obj"

def do_drop(table_name, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return "Milvus doesn't have a collection named {}".format(table_name)
        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        logger.error(" Error with  drop table: {}".format(e))