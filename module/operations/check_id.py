import sys

from loguru import logger

def checkid_mysql(table_name, person_id, mysql_cli):
    try:
        status = mysql_cli.checkid(table_name,person_id)
        if status == None:
            return False
        else:
            return True
    except Exception as e:
        logger.error("Error with find person id {}".format(e))