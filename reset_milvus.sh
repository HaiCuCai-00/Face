sudo docker-compose down
sudo rm -rf volumes
sudo docker-compose up -d
cd module/operations/
python3 reset_milvus.py
