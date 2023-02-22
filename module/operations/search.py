import sys

from loguru import logger

# sys.path.append("../..")
# from configs.config import DEFAULT_TABLE
DEFAULT_TABLE="milvus_obj"

def do_search(table_name, feat, top_k, masked, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        if not milvus_client.has_collection(table_name):
            return "no_table" ,None,None
        vectors = milvus_client.search_vectors(table_name, feat, top_k)
        ids = [str(x.id) for x in vectors[0]]
        if len(ids) == 0:
            return None, None
        person_id, image_path = mysql_cli.search_by_milvus_ids(ids, table_name)
        distances = [x.distance for x in vectors[0]]
        print("person_id: ",person_id)
        # print(distances)
        print("path: ", image_path)
        person_ids = ids
        return person_id, distances, image_path, person_ids
    #return ids,vectors
    except Exception as e:
        logger.error(" Error with search : {}".format(e))