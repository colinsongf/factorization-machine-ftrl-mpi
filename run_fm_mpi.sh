#!/bin/bash

#tt=`date`
#mkdir backup/"$tt"
#mv train backup/"$tt"
#mv *.log backup/"$tt"
#mv core backup/"$tt"
#make
#rm log/*
/home/xiaoshu/hadoop_job/bin/mpirun -np 4 ./train /home/xiaoshu/hadoop_job/newsToNewsModel/relative_news/gbdt_click/feature/onehot_encoding/libsvm_train_onehotencoding.data  /home/xiaoshu/hadoop_job/newsToNewsModel/relative_news/gbdt_click/feature/onehot_encoding/libsvm_test_onehotencoding.data
#/home/xiaoshu/hadoop_job/bin/mpirun -np 2 ./train ./data/agaricus.txt.train ./data/agaricus.txt.test
