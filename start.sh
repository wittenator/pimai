#!/bin/bash

finish()
{
    CURRENT_UID=$(id -u):$(id -g) docker-compose stop
    exit
}
trap finish SIGINT

CURRENT_UID=$(id -u):$(id -g) docker-compose up --build -d

time=$(date +"%s")
until docker logs --since $time VAENotebook 2>&1 | grep -m 1 "127.0.0.1"; do sleep 5 ; done
token=$(docker logs --since $time VAENotebook 2>&1 | grep '127.0.0.1' | grep -m 1 -oP 'token=\K(.*)')
xdg-open http://localhost:8888/?token=$token

while :; do
    sleep 5
done