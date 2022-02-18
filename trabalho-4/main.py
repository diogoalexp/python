#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2
from PIL import Image
#===============================================================================

INPUT_IMAGE =  'exemplos/205.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 10

#===============================================================================

def estimativa (img, kernel, ruido):
    output_erosion = cv2.erode(img, kernel)
    
    estimativa_padrao(output_erosion, ruido)
    
    return img

def estimativa_padrao (img, ruido):

    label_image = img.copy()
    label_count = 0
    contador = 0
    rows, cols = label_image.shape
    for j in range(rows):
        for i in range(cols):
            pixel = label_image[j, i]
            if 255 == pixel:
                label_count += 1
                count = cv2.floodFill(label_image, None, (i, j), label_count)
                if count[0] > 1:
                    contador += 1

    if(ruido):
        print(label_count)
        # print("estimativa_padrao: TOTAL:", label_count)    
    else:
        print(contador)
        # print("estimativa_padrao: TOTAL(Sem Ruido):", contador)    

#-------------------------------------------------------------------------------
def estimativa_bordas (img):

    # image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    image = img.copy()

    M, N = image.shape

    # remove the blobs that touch the edges
    for i in range(M):
        for j in [0, N-1]:
            if image[i, j] == 255:
                cv2.floodFill(image, None, (j, i), 0)
    for i in [0, M-1]:
        for j in range(N):
            if image[i, j] == 255:
                cv2.floodFill(image, None, (j, i), 0)

    # cv2.imshow("X1 TESTE", image)
    # cv2.imwrite ('X1 - test.png', image*255)                
                
    # count every blob
    n_objects = 0
    for i in range(M):
        for j in range(N):
            if image[i, j] == 255:
                n_objects += 1
                cv2.floodFill(image, None, (j, i), n_objects)

    # cv2.imshow("X2 TESTE", image)
    # cv2.imwrite ('X2 - test.png', image*255)                  

    print("estimativa_bordas: TOTAL:", n_objects)    


#===============================================================================

def main ():
    sys.setrecursionlimit(150000)

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    # img = binariza (img, THRESHOLD)
    if img is None:
        print ('Erro abrindo a imagem depois do binariza.\n')
        sys.exit ()

    cv2.imshow ('00 - binarizada', img)
    cv2.imwrite ('00 - binarizada.png', img*255)

    
    image = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    # thresh, output_binthresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # print("Fixed threshold", thresh)
    # cv2.imshow("Binary Threshold (fixed)", output_binthresh)

    # thresh, output_otsuthresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print("Otsu threshold", thresh)
    # cv2.imshow("Binary Threshold (otsu)", output_otsuthresh)

    output_adapthresh = cv2.adaptiveThreshold (image, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -20.0)
    # cv2.imshow("output_adapthresh", output_adapthresh)
    # cv2.imwrite ('01 - output_adapthresh.png', output_adapthresh*255)

    # output_adapthresh = img


    kernel = np.ones((7,7),np.uint8)

    output_erosion = cv2.erode(output_adapthresh, kernel)
    cv2.imshow("output_erosion", output_erosion)
    cv2.imwrite ('02 - output_erosion.png', output_erosion*255)


    output_dilation = cv2.dilate(output_adapthresh, kernel)
    cv2.imshow("output_dilation", output_dilation)
    cv2.imwrite ('03 - output_dilation.png', output_dilation*255)

    output_erosion_plus_dilation = cv2.erode(output_adapthresh, kernel)
    output_erosion_plus_dilation = cv2.dilate(output_erosion_plus_dilation, kernel)
    cv2.imshow("output_erosion_plus_dilation", output_erosion_plus_dilation)
    cv2.imwrite ('04 - output_erosion_plus_dilation.png', output_erosion_plus_dilation*255)

    output_dilation_plus_erosion = cv2.erode(output_adapthresh, kernel)
    output_dilation_plus_erosion = cv2.dilate(output_dilation_plus_erosion, kernel)
    cv2.imshow("output_dilation_plus_erosion", output_dilation_plus_erosion)
    cv2.imwrite ('05 - output_dilation_plus_erosion.png', output_dilation_plus_erosion*255)

    output_morph_open = cv2.morphologyEx(output_adapthresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("output_morph_open", output_morph_open)
    cv2.imwrite ('06 - output_morph_open.png', output_morph_open*255)

    output_morph_close = cv2.morphologyEx(output_adapthresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("output_morph_close", output_morph_close)
    cv2.imwrite ('07 - output_morph_close.png', output_morph_close*255)

    output_morph_ellipse = cv2.morphologyEx(output_adapthresh, cv2.MORPH_ELLIPSE, kernel)
    cv2.imshow("output_morph_ellipse", output_morph_ellipse)
    cv2.imwrite ('08 - output_morph_ellipse.png', output_morph_ellipse*255)

    output_gradient = cv2.morphologyEx(output_adapthresh, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("output_gradient", output_gradient)
    cv2.imwrite ('09 - output_gradient.png', output_gradient*255)

    output_tophat = cv2.morphologyEx(output_adapthresh, cv2.MORPH_TOPHAT, kernel)
    cv2.imshow("output_tophat", output_tophat)
    cv2.imwrite ('10 - output_tophat.png', output_tophat*255)    

    output_morph_erode = cv2.morphologyEx(output_adapthresh, cv2.MORPH_ERODE, kernel)
    cv2.imshow("output_morph_erode", output_morph_erode)
    cv2.imwrite ('11 - output_morph_erode.png', output_morph_erode*255)    

    output_morph_dilatation = cv2.morphologyEx(output_adapthresh, cv2.MORPH_DILATE, kernel)
    cv2.imshow("output_morph_dilatation", output_morph_dilatation)
    cv2.imwrite ('08 - output_morph_dilatation.png', output_morph_dilatation*255)  

    print('TESTES:')
    print('')

    # print('output_erosion')
    # estimativa_padrao(output_erosion)
    # estimativa_bordas(output_erosion)

    # print('output_dilation')
    # estimativa_padrao(output_dilation)
    # estimativa_bordas(output_dilation)   

    # print('output_erosion_plus_dilation')
    # estimativa_padrao(output_erosion_plus_dilation)
    # estimativa_bordas(output_erosion_plus_dilation)  

    # print('output_dilation_plus_erosion')
    # estimativa_padrao(output_dilation_plus_erosion)
    # estimativa_bordas(output_dilation_plus_erosion)     

    # print('output_morph_open')
    # estimativa_padrao(output_morph_open)
    # estimativa_bordas(output_morph_open)        

    # print('output_morph_close')
    # estimativa_padrao(output_morph_close)
    # estimativa_bordas(output_morph_close)    

    # print('output_morph_ellipse')
    # estimativa_padrao(output_morph_ellipse)
    # estimativa_bordas(output_morph_ellipse)      

    # print('output_gradient')
    # estimativa_padrao(output_gradient)
    # estimativa_bordas(output_gradient)        

    # print('output_tophat')
    # estimativa_padrao(output_tophat)
    # estimativa_bordas(output_tophat) 

    # print('output_morph_erode')
    # estimativa_padrao(output_morph_erode)
    # estimativa_bordas(output_morph_erode)    

    # print('output_morph_dilatation')
    # estimativa_padrao(output_morph_dilatation)
    # estimativa_bordas(output_morph_dilatation)      
            
    print("ESTIMATIVA")
    for j in [5,7,9]:
        for i in [5,7,9]:
            #  print("kernel:",j ,i)
             estimativa(output_adapthresh, np.ones((j,i),np.uint8), False)
   
    # start_time = timeit.default_timer ()
    # print ('Tempo: %f' % (timeit.default_timer () - start_time))
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================

