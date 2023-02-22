import os

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "0.0.0.0")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 512))
INDEX_FILE_SIZE = int(os.getenv("INDEX_FILE_SIZE", 1024))
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus_obj")
TOP_K = int(os.getenv("TOP_K", 10))

############### Number of log files ###############
LOGS_NUM = int(os.getenv("logs_num", 0))
SCORE =int(os.getenv("Distance",500))

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")
