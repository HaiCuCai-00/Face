import sys
import os
import shutil
sys.path.append("..")
from mysql_helpers import MySQLHelper

if __name__ == "__main__":
    mysql = MySQLHelper()
    collection_name ='_2090App'
    path = "/media/ai-r-d/DATA1/Face_triton/service"
    image_paths = mysql.show_data(collection_name)
    for i in image_paths:
        source = os.path.join(path,i[1])
        destination = source.replace("static","face_images")
        shutil.copyfile(source, destination)