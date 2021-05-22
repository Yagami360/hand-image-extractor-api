#!/bin/sh
#set -eu
#CONTAINER_NAME=graphonomy_server_cpu_container

docker-compose stop
docker-compose up -d
#docker exec -it -u $(id -u $USER):$(id -g $USER) ${CONTAINER_NAME} bash
