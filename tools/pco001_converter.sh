#!/bin/bash

for dataset in banana iris messidor_features; do
    python convert_csv_npy.py -i ../data/pco001/csv/${dataset}.csv
    python convert_npy_dat.py -i ../data/pco001/csv/${dataset}.npy -l
    mv ../data/pco001/csv/${dataset}.dat ../data/pco001/
    rm -f ../data/pco001/csv/${dataset}.npy
done