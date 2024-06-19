import cv2
import os
import pytesseract
import numpy as np

#  Ruta de la imagen
folder_path = os.path.join('Matrix_Reader', 'Imagenes_Pruebas')
image_name = 'Imagen_pruebas5.png'
image_path = os.path.join(folder_path, image_name)


#  Cargar la imagen y establecerla en escalas de grices
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#  Extraemos los contornos o umbralizamos la imagen (thresholding)
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

_, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
b_img = cv2.bitwise_not(cv2.bitwise_xor(img_gray, vertical_horizontal_lines))


cv2.imshow('', vertical_horizontal_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()



