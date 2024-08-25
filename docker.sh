docker build -t pinns_cuda_11.8:latest -f dockerfile .
docker run --gpus all -it --rm --name pinns \
--mount type=bind,src=$PWD,dst=/mnt \
pinns_cuda_11.8:latest