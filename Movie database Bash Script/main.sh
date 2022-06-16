#!/bin/bash

bash menu.sh
read opcio

case $opcio in
1)
    bash opcio1.sh
    bash main.sh
;;
2)
    bash opcio2.sh
    bash main.sh
;;
3)
    bash opcio3.sh
    bash main.sh
;;
4)
    clear
    exit
;;
*)
    clear
    echo Error: Opció $opcio no vàlida
    echo Esperi 3 segons per continuar
    sleep 1
    clear
    echo Error: Opció $opcio no vàlida
    echo Esperi 2 segons per continuar
    sleep 1
    clear
    echo Error: Opció $opcio no vàlida
    echo Esperi 1 segon per continuar
    sleep 1
    bash main.sh
esac
