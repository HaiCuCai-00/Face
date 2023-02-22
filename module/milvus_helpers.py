from shutil import ExecError
import sys
from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType,FieldSchema,connections, utility

# sys.path.append("..")
# from configs.config import METRIC_TYPE, MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION
METRIC_TYPE="L2"
MILVUS_HOST="0.0.0.0"
MILVUS_PORT=19530
VECTOR_DIMENSION=512

class MilvusHelper:
    def __init__(self):
        try:
            self.collection=None
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            logger.debug(" Successfully connect to MiLvus with IP: {} and Port: {}".format(MILVUS_HOST, MILVUS_PORT))
        except Exception as e:
            logger.debug("Faild to load data to Milvus {}".format(e))
            sys,exit(1)
    def set_collection(self, collection_name):
        try:
            if self.has_collection(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                raise Exception(
                    "There has no collection named:{}".format(collection_name)
                )
        except Exception as e:
            logger.error("Failed to load data to Milvus: {}".format(e))
            sys.exit(1)

    def has_collection(self, collection_name):
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            logger.debug()

    def create_collection(self, collection_name):
        try:
            if not self.has_collection(collection_name):
                print(collection_name)
                #print(collection_name)
                field1 = FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    descrition="int64",
                    is_primary=True,
                    auto_id=True,
                )
                field2 = FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    descrition="float vector",
                    dim=512,
                    is_primary=False,
                )
                field3 = FieldSchema(
                    name="masked", dtype=DataType.BOOL, description="boolean"
                )
                schema = CollectionSchema(
                    fields=[field1, field2,field3],
                    description="collection description",
                )
                self.collection = Collection(name=collection_name, schema=schema)
                logger.debug("Create Milvus collection: {}".format(self.collection))
            return "OK"
        except Exception as e:
            logger.error("Failed to load data to Milvus {}".format(e))
            sys.exit(1)
    
    def insert(self, collection_name, vectors):
        try:
            self.create_collection(collection_name)
            collection = Collection(collection_name)
            mr=collection.insert(vectors)
            ids = mr.primary_keys
            #self.collection.load()
            logger.debug(
                "Insert vectors to Milvus in collection: {} with {} rows in collection: {}".format(
                    collection_name, len(vectors), collection_name
                )
            )
            #print(ids)
            return ids
        except Exception as e:
            logger.error("Failed to load data to Milvus: {}".format(e))
            sys.exit(1)

    def create_index(self, collection_name):
        try:
            self.set_collection(collection_name)
            default_index = {
                "index_type": "IVF_FLAT",
                "metric_type": METRIC_TYPE,
                "params": {"nlist": 16384},
            }
            status = self.collection.create_index(
                field_name="embedding", index_params=default_index
            )
            if not status.code:
                logger.debug(
                    "Successfully create index in collection:{} with param:{}".format(
                        collection_name, default_index
                    )
                )
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            logger.error("Failed to create index: {}".format(e))
            sys.exit(1)

    def delete_collection(self, collection_name):
        try:
            self.set_collection(collection_name)
            collection=Collection(collection_name)
            collection.drop()
            logger.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            logger.error("Failed to drop collection")
            sys.exit(1)
    
    def delete(self, collection_name, ids):
        try:
            for i in ids:
                # expr = "id in " + str(list(map(int, ids)))
                expr = "id in [" + str(i) + "]"
                print("expr: ",expr)
                # expr = "id in [428822398174756872]"
                self.set_collection(collection_name)
                res = self.collection.delete(expr)
                logger.debug(
                    "Successfully delete entities: {} in collection: {}".format(
                        res, collection_name
                    )
                )
            return "ok" 
        except Exception as e:
            self.collection.release()
            self.collection.load()
            # NOTE: This function will cause error but the data with deleted
            logger.warning("This function will cause error but the datas are deleted")
            logger.error("Failed to drop collection: {}".format(e))

    def search_vectors(self, collection_name,vectors, top_k):
        try:
            self.set_collection(collection_name)
            search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
            # data = [vectors]
            collection=Collection(collection_name)
            collection.load()
            res = collection.search(
                vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                #expr=f"masked == {1 if masked else 0}",
            )
            # print(res[0])
            logger.debug("Successfully search in collection: {}".format(res))
            return res
        except Exception as e:
            logger.error("Failed to search vectors in Milvus: {}".format(e))

    def count(self, collection_name):
        try:
            self.set_collection(collection_name)
            collection=Collection(collection_name)
            num=collection.num_entities
            logger.debug("Successfully get the num: {} of the collection: {}".format(num, collection_name))
            return num
        except Exception as e:
            logger.error("Failed to count vector in Milvus: {}".format(e))
            sys.exit(1)

if __name__ =="__main__":
    mivlvus=MilvusHelper()
    collection_name="_2090App"
    ids='436952627349291148'
    # mivlvus.delete_collection(collection_name)
    # mivlvus.create_collection(collection_name)
    # mivlvus.delete(collection_name,ids)
    # a=mivlvus.delete_collection(collection_name)
    # print(a)
    #mivlvus.insert(collection_name,[vector,[False]])
    #mivlvus.search_vectors(collection_name,vector,1)
    mivlvus.count(collection_name)
