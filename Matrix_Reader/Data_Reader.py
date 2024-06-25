import cv2
import os
import pytesseract
import numpy as np
import pandas as pd


#  Ruta de la imagen
folder_path = os.path.join('Matrix_Reader', 'Imagenes_Pruebas')
image_name = 'Imagen_pruebas.png'
image_path = os.path.join(folder_path, image_name)


#  Cargar la imagen y establecerla en escalas de grices
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#  Binarizamos la img-agen en escala de grises (0 Black, 255 White)
_, img_binary_otsu = cv2.threshold(255-img_gray, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#  Invertimos la imagen con 255-img_blur


#  Extraemos las lineas verticales 
#  Kernel para la detección de lineas
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1] // 30))  
#  Aplicamos erosión y dilatación, para la separación de pixeles unidos y resaltar los que no 
eroded_img = cv2.erode(img_binary_otsu, vertical_kernel, iterations=3)
vertical_lines = cv2.dilate(eroded_img, vertical_kernel, iterations=3)


#  Extraemos las líneas horizontales
#  Proceso analogo al de la la líneas verticales
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 30, 1))
eroded_img2 = cv2.erode(img_binary_otsu, horizontal_kernel, iterations=3)
horizontal_lines = cv2.dilate(eroded_img2, horizontal_kernel, iterations=3)


#  Combinamos las lineas veticales y horizontales
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) 
vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=2)

#  Aplicamos operaciones bit a bit a la imagen en escala de grises y los contornos combinados 
_, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
b_img = cv2.bitwise_not(cv2.bitwise_xor(img_gray, vertical_horizontal_lines))


#  Extraemos los contornos de cada celda de la tabla
contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#  Craeamos delimitadores para cada respectivo contorno o rectangulo
boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key= lambda x: x[1][1]))

# Almacenamos los contornos y los dibujamos en la imagen original
boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 1000 and h < 500:
        image = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        boxes.append([x, y, w, h])
 
# almacenamos las filas y columnas para combinarlas posteriormente
rows = []
colums = []
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
mean = np.mean(heights)
colums.append(boxes[0])
previous = boxes[0]
for i in range(1, len(boxes)):
    if boxes[i][1] <= previous[1]+mean/2:
        colums.append(boxes[i])
        previous = boxes[i]
        if i == len(boxes)-1:
            rows.append(colums)
    else:
        rows.append(colums)
        colums = []
        previous = boxes[i]
        colums.append(boxes[i])

#  Encontramos el total de celdas en cada fila 
total_cells = 0
for i in range(len(rows)):
    if len(rows[i]) > total_cells:
        total_cells = len(rows[i])

#  Encontramos el centro de las celdas  
center = [int(rows[i][j][0] + rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
center = np.array(center)
center.sort()

#  Creamos una lista con las coordenas de cada caja
boxes_list = []
for i in range(len(rows)):
    l = []
    for k in range(total_cells):
        l.append([])
    for j in range(len(rows[i])):
        diff = abs(center - (rows[i][j][0] + rows[i][j][2]/4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        l[indexing].append(rows[i][j])
    boxes_list.append(l)

#  Extraemos la información de las celdas
#  Se debe cambiar la ruta de tesseract dependiendo el lugar de la instalación de este
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
dataframe_final = []
for i in range(len(boxes_list)):
    for j in range(len(boxes_list[i])):
        s = ''
        if len(boxes_list[i][j]) == 0:
            dataframe_final.append(' ')
        else:
            for k in range(len(boxes_list[i][j])):
                y, x, w, h = boxes_list[i][j][k][0], boxes_list[i][j][k][1], boxes_list[i][j][k][2], boxes_list[i][j][k][3]
                roi = b_img[x:x+h, y:y+w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations=1)
                erosion = cv2.erode(dilation, kernel, iterations=2)
                out = pytesseract.image_to_string(erosion, config=r'--oem 3 --psm 6')
                if len(out) == 0:
                    out = pytesseract.image_to_string(erosion)
                s = s +" "+ out.strip()
            
            dataframe_final.append(s.strip())


#  Crear Array con los datos
array = np.array(dataframe_final)
#  Creamos el dataframe
dataframe = pd.DataFrame(array.reshape(len(rows), total_cells))
data = dataframe.style.set_properties(align='left')
print(dataframe)

#  Imagen con el contorno de cada celda
cv2.imshow('', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Aquí se muestra como resolver dicho error, en la celda [1][0]
#  Aún así con el error, la disminución de tiempo es considerable a digitar cada dato manualmente
dataframe[1][0] = '10,2'
dataframe.drop(axis=1, labels=5, inplace=True)
print(dataframe)