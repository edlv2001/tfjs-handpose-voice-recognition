#!/bin/zsh

# Directorio donde se encuentran los archivos .wav
dir_base="data/audio"

# Iterar sobre cada archivo .wav en el directorio
for file in "$dir_base"/*[0-9][0-9].wav; do
    # Extraer los últimos dos dígitos del nombre del archivo
    dir_name="${file##*-}"
    dir_name="${dir_name%%.*}"

    # Crear el directorio si no existe
    if [[ ! -d "$dir_base/$dir_name" ]]; then
        mkdir "$dir_base/$dir_name"
    fi

    # Mover el archivo al directorio correspondiente
    mv "$file" "$dir_base/$dir_name/"
done
