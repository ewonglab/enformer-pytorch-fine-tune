#!/usr/bin/env bash
#
#
#

for cell_type in ./best_result/*
do
#  echo "$cell_type"
  cell_name=$(echo "$cell_type" | cut -d '/' -f 3)
#  echo "$cell_name"

  new_dir="./result_to_paola/${cell_name}/"
  mkdir -p "$new_dir"


  if test $cell_name = 'mixed_meso'
  then
    cp "${cell_type}/Tanh_AUROC_PLOT.csv" "${new_dir}/AUROC.csv"
  else
    cp "${cell_type}/AUROC_PLOT.csv" "${new_dir}/AUROC.csv"
  fi 

done
