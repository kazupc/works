#!/bin/bash
case "$1" in
    "1") MODEL=/home/ec2-user/chainer-fast-neuralstyle/models/composition.model
    ;;
    "2") MODEL=/home/ec2-user/chainer-fast-neuralstyle/models/seurat.model
    ;;
    "3") MODEL=/home/ec2-user/chainer-fast-neuralstyle/models/gogh.model 
    ;;
esac
sudo -u root /home/ec2-user/anaconda3/bin/python /home/ec2-user/chainer-fast-neuralstyle/generate.py /var/www/html/img/input.jpg -m "$MODEL" -o /var/www/html/img/output.jpg
