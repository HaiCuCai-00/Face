import sys

from loguru import logger

# sys.path.append("../..")
# from configs.config import DEFAULT_TABLE
DEFAULT_TABLE="milvus_obj"

def do_count(table_name,milvus_cli, mysql_cli):
    print (table_name)
    print (type(table_name))
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None
        num = mysql_cli.count_table(table_name)
        return num
        
    except Exception as e:
        logger.error(" Error with count table {}".format(e))
