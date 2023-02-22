import sys

from loguru import logger

# sys.path.append("../..")
# from configs.config import DEFAULT_TABLE
DEFAULT_TABLE="milvus_obj"

# Combine the id of the vector and the name of the image into a list
def format_data(ids, person_ids, image_paths):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), person_ids[i], image_paths)
        data.append(value)
    return data


def do_load(table_name, vectors, person_ids, image_paths, mil_cli, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        ids = mil_cli.insert(table_name, vectors)
        mil_cli.create_index(table_name)
        mysql_cli.create_mysql_table(table_name)
        mysql_cli.load_data_to_mysql(
            table_name, format_data(ids, person_ids, image_paths)
        )
        return ids
    except Exception as e:
        logger.error(" Error with insert : {}".format(e))
