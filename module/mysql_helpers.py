import sys

import pymysql
from loguru import logger

# sys.path.append("..")
# from configs.config2 import MYSQL_DB, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD
MYSQL_DB="mysql"
MYSQL_HOST="127.0.0.1"
MYSQL_PORT=3306
MYSQL_USER="root"
MYSQL_PWD="123456"

class MySQLHelper:
    def __init__(self):
        self.conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port=MYSQL_PORT,
            password=MYSQL_PWD,
            database=MYSQL_DB,
            local_infile=True,
        )
        self.cursor = self.conn.cursor()

    # Test the connection, and will reconnect if disconnected
    def test_connection(self):
        try:
            self.conn.ping()
        except Exception:
            self.conn = pymysql.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                port=MYSQL_PORT,
                password=MYSQL_PWD,
                database=MYSQL_DB,
                local_infile=True,
            )
            self.cursor = self.conn.cursor()

    # Create mysql table if not exists
    def create_mysql_table(self, table_name):
        self.test_connection()
        sql = (
            "create table if not exists "
            + table_name
            + "(milvus_id TEXT, person_id TEXT, image_path TEXT );"
        )
        try:
            self.cursor.execute(sql)
            logger.debug("MYSQL create table: {} with sql: {}".format(table_name, sql))
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))

    # Batch insert (Milvus_ids, person_id, img_path) to mysql
    def load_data_to_mysql(self, table_name, data):
        self.test_connection()
        sql = (
            "insert into "
            + table_name
            + " (milvus_id, person_id, image_path) values (%s,%s,%s);"
        )
        try:
            self.cursor.executemany(sql, data)
            self.conn.commit()
            logger.debug(
                "MYSQL loads data to table: {} successfully".format(table_name)
            )
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))

    # Search data by person_id
    def search_data_by_person_id(self, table_name, data):
        self.test_connection()
        sql = "select milvus_id  from " + table_name + " where person_id = %s ;"
        try:
            self.cursor.execute(sql, data)
            results = self.cursor.fetchall()
            results = [res[0] for res in results]
            logger.debug("MYSQL search by person_id: {}".format(data))
            return results
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))
    
    def show_image(self, table_name, person_id):
        self.test_connection()
        sql ="select image_path from "+ table_name +" where person_id = %s ;"
        try:
            self.cursor.execute(sql,person_id)
            results = self.cursor.fetchall()
            res= results[0]
            logger.debug("Mysql search by image: {}".format(person_id))
            return res
        except Exception as e:
            logger.error("Mysql error: {} with sql: {}".format(e,sql)) 

    # Get the person_id according to the milvus ids
    def search_by_milvus_ids(self, ids, table_name):
        self.test_connection()
        str_ids = str(ids).replace("[", "").replace("]", "")
        sql = (
            "select person_id, image_path from "
            + table_name
            + " where milvus_id in ("
            + str_ids
            + ");"
        )
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            id_person=[]
            image_path=[]
            for res in results:
                print(res)
                id_person.append(res[0])
                image_path.append(res[1])
            print("path: ",image_path)
            logger.debug("MYSQL search by milvus id.")
            return id_person ,image_path
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))

    # Delete mysql table if exists
    def delete_table(self, table_name):
        self.test_connection()
        sql = "drop table if exists " + table_name + ";"
        try:
            self.cursor.execute(sql)
            logger.debug("MYSQL delete table:{}".format(table_name))
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))

    # Delete entities by the ids
    def delete(self, table_name, ids):
        self.test_connection()
        str_ids = str(ids).replace("[", "").replace("]", "")
        sql = "delete from " + table_name + " where milvus_id in (" + str_ids + ");"
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            logger.debug("MYSQL delete id:{} in table:{}".format(ids, table_name))
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))


    def delete_person(self, table_name, ids):
        self.test_connection()
        sql = "delete from " + table_name + " where person_id = " + ids + " ;"
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            logger.debug("MYSQL delete id:{} in table:{}".format(ids, table_name))
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))
    
    def show_data(self, table_name):
        self.test_connection()
        sql= "select person_id,image_path from "+ table_name + " ;"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            logger.debug("MYSQL Show table:{}".format(table_name))
            return results
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))


    # Delete all the data in mysql table
    def delete_all_data(self, table_name):
        self.test_connection()
        sql = "delete from " + table_name + ";"
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            logger.debug("MYSQL delete all data in table:{}".format(table_name))
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))

    # Get the number of mysql table
    def count_table(self, table_name):
        self.test_connection()
        sql = "select count(milvus_id) from " + table_name + ";"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            logger.debug("MYSQL count table:{}".format(table_name))
            return results[0][0]
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))

    def checkid(self,table_name,ids):
        self.test_connection()
        self.create_mysql_table(table_name)
        sql="select * from "+ table_name +" where person_id = %s ;"
        try:
            self.cursor.execute(sql,ids)
            results=self.cursor.fetchall()
            print(results[0][0])
            return results[0][0]
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e,sql))

    def upadte_milvus_id(self,table_name, milvus_id, path):
        self.test_connection()
        self.create_mysql_table(table_name)
        sql = f"UPDATE {table_name} SET milvus_id = {milvus_id} WHERE image_path = '{path}'"
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            logger.debug(
                "MYSQL loads data to table: {} successfully".format(table_name)
            )
        except Exception as e:
            logger.error("MYSQL ERROR: {} with sql: {}".format(e, sql))


if __name__=="__main__":
    mysql=MySQLHelper()
    ids='bach'
    table='_2090App'
    #mysql.delete_person(table,ids)
    #mysql.delete(table,ids)
    #b=mysql.show_data(table)
    #print(b)
    a=mysql.count_table(table)
    print(a)
    # milvus_id=mysql.search_data_by_person_id(table,ids)
    # print(milvus_id)
    # ids=mysql.checkid(table,ids)
    # print(ids)
