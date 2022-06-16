#!/bin/bash

clear

echo --------------------------------------
echo -e '\t   'Llistar per any
echo -e '--------------------------------------\n'
echo Introdueixi un any:
read year
echo

awk -v year="$year" -F "," '$5 == year' netflix.csv | cut -d, -f 1,2 --output-delimiter=', '

echo 
echo Premi enter per continuar
read ret