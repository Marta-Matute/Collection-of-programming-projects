#!/bin/bash

clear

echo --------------------------------------
echo -e '\t  'Recomanació ràpida
echo -e '--------------------------------------\n'

n_lines=`wc -l < netflix.csv`
film_line=`shuf -i 2-$n_lines -n 1`
title=`head -$film_line netflix.csv | tail -1 | cut -d, -f1`
year=`head -$film_line netflix.csv | tail -1 | cut -d, -f5`
rating=`head -$film_line netflix.csv | tail -1 | cut -d, -f2`
description=`head -$film_line netflix.csv | tail -1 | cut -d, -f3`

echo $title, $year
echo $rating
echo $description

echo 
echo Premi enter per continuar
read ret

