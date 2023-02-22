import sys 
from loguru import logger

DEFAULT_TABLE="_2090App"

def do_show(table_name, person_id, mysql_cli, milvus_client):
    try:
        if not table_name:
            table_name=DEFAULT_TABLE
        if not milvus_client.has_collection(table_name):
            return "no_table" 
        image_path=mysql_cli.show_image(table_name, person_id)
        return image_path
    except Exception as e:
        logger.error("Error with show: {}".format(e))
        