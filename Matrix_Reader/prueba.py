import cv2
import os
import pytesseract
import numpy as np

#  Ruta de la imagen
folder_path = os.path.join('Matrix_Reader', 'Imagenes_Pruebas')
image_name = 'Imagen_pruebas.png'
image_path = os.path.join(folder_path, image_name)

#  Cargar la imagen y establecerla en escalas de grices
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#  Extraemos los contornos o umbralizamos la imagen (thresholding)
thresh, img_binary_otsu = cv2.threshold(255-img_gray, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


#  Extraemos las lineas verticales y aplicamos efectos a dichos contornos
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1] // 150))
eroded_img = cv2.erode(img_binary_otsu, vertical_kernel, iterations=5)
vertical_lines = cv2.dilate(eroded_img, vertical_kernel, iterations=5)


#  Extraemos las líneas horizontales y aplicamos efectos a dichos contornos
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 150, 1))
eroded_img2 = cv2.erode(img_binary_otsu, horizontal_kernel, iterations=5)
horizontal_lines = cv2.dilate(eroded_img2, horizontal_kernel, iterations=5)

#  Sumamos las lineas extraidas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#  Adicionamos o mezclamos las lineas verticales y horizontales
vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)

thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
b_img = cv2.bitwise_not(cv2.bitwise_xor(img_gray, vertical_horizontal_lines))


cv2.imshow('', img_binary_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()



