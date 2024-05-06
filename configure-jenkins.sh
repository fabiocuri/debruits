#!/bin/bash

set -e

# Launch Jenkins
docker run -p 8080:8080 -p 50000:50000 -d -v jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock jenkins/jenkins:lts
sleep 20
JENKINS_CONTAINER_ID=$(docker ps | grep jenkins | awk '{print $1}')
docker exec -u 0 -it $JENKINS_CONTAINER_ID bash
sleep 3
curl https://get.docker.com/ > dockerinstall && chmod 777 dockerinstall && ./dockerinstall
chmod 666 /var/run/docker.sock