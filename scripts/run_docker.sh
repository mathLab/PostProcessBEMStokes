#!/bin/bash
IMG=`grep image .gitlab-ci.yml | awk '{print $2}'` 
echo $IMG

CMD=`grep -e '- ' .gitlab-ci.yml | sed 's/- //'`
docker image rm -f $IMG
docker pull $IMG 
docker run -v `pwd`/..:/builds/ $IMG /bin/sh -c "$CMD" 
