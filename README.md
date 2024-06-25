El objetivo de este proyecto es agilizar la transcricción o necesidad de digitar manualmente datos estructurados (tablas) almacenados en imaganes png o jpg. Posteriormente estos datos se almacenan en DataFrame de pandas para su debido tratamiento.

Librerías utilizadas:
Para este proyecto se utilizaron las siguientes librerias

    * cv2
    * os
    * numpy
    * pandas
    * pytesseract

Principalmente se utiliza cv2 para la lectura y procesamiento de las imagenes, posteriormente con la imagen procesada utilizamos pytesseract para la extración de los caracteres.


El programa aun tiene algunos bugs, como alguna de las celdas que no son detectadas por cv2, o la reconocimiento de caracteres que no están presentes por tesseract. Estos problemas disminuyen en función de la calidad de la imagen, pero en terminos generales no suele haber más de un error en la extracción de los datos, cumpliendo casí por completo con el obetivo del proyecto.


Por ultimo, los merecidos creditos al desarrollador del cual tome parte del codigo para este proyecto.

Github : Soumi7

Link al codigo, https://github.com/Soumi7/Table_Data_Extraction/blob/main/medium_table.ipynb?source=post_page-----5a2934f61caa--------------------------------
