#!/bin/zsh

# Directorio base
dir_base="data/trainAudio"
# Array para almacenar los objetos JSON
json_array=()
# Contador para classNumber
class_number=0

# Leer cada subdirectorio en el directorio base
for dir in "$dir_base"/*(/); do
    # Obtener el nombre del subdirectorio para className
    class_name="${dir:t}"

    # Leer cada archivo .wav en el subdirectorio
    for file in "$dir"/*.wav; do
        # Obtener el nombre del archivo
        file_name="${file:t}"
        
        # Crear el objeto JSON con formato y tabulaciones adicionales
        json_object=$(
            printf "\t{\n"
            printf "\t\t\"fileName\": \"%s\",\n" "$class_name/$file_name"
            printf "\t\t\"className\": \"%s\",\n" "$class_name"
            printf "\t\t\"classNumber\": %d,\n" "$class_number"
            printf "\t\t\"fold\": \"5\"\n"
            printf "\t}"
        )
        
        # Añadir el objeto al array
        json_array+=("$json_object")
    done

    # Incrementar el classNumber para el siguiente subdirectorio
    ((class_number++))
done

# Unir los elementos del array en una cadena, separados por comas y saltos de línea
json_elements=$(printf ",\n%s" "${json_array[@]}")
# Eliminar la primera coma para evitar un elemento vacío al inicio
json_elements=${json_elements:2}

# Formatear como un array JSON
json_result=$(printf "[\n%s\n]" "$json_elements")

# Imprimir el resultado
echo $json_result
