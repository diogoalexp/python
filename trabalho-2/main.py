#===============================================================================
# Trabalho 2
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'b01 - Original.bmp'
KSIZE = (7, 7)

#===============================================================================

def blurIngenuo (img, ksize):
  
    h, w = ksize
    margenVertical = int(h/2)
    margenHorizontal = int(w/2)
    altura, largura, _ = img.shape

    resultado = np.zeros(img.shape)

    for y in range(margenVertical, altura-margenVertical):             #para cada linha y
        for x in range(margenHorizontal, largura-margenHorizontal):            #para cada coluna x
            soma = 0
            for y2 in range(y-margenVertical, y+margenVertical+1):            #para cada linha da janela y
                for x2 in range(x-margenHorizontal, x+margenHorizontal+1):            #para cada coluna da janela x
                    soma = soma + img[y2, x2]
            resultado[y][x] = soma / (h*w)

    return resultado

#-------------------------------------------------------------------------------
def blurSeparavel (img, ksize):

    h, w = ksize
    margenVertical = int(h/2)
    margenHorizontal = int(w/2)
    altura, largura, _ = img.shape
    
    separavel = np.zeros(img.shape)
    resultado = np.zeros(img.shape)


    for y in range(margenVertical, altura-margenVertical):             #para cada linha y
        for x in range(margenHorizontal, largura-margenHorizontal):            #para cada coluna x
            soma = 0
            for y2 in range(y-margenVertical, y+margenVertical+1):                        
                    soma = soma + img[y2][x]
            separavel[y][x] = soma / (h)    

    for y in range(margenVertical, altura-margenVertical):             #para cada linha y
        for x in range(margenHorizontal, largura-margenHorizontal):            #para cada coluna x
            soma = 0
            for x2 in range(x-margenHorizontal, x+margenHorizontal+1):                        
                    soma = soma + separavel[y][x2]
            resultado[y][x] = soma / (w)    

    return resultado

#-------------------------------------------------------------------------------

def blurIntegrais (img, ksize):

    h, w = ksize
    margenVertical = int(h/2)
    margenHorizontal = int(w/2)    
    altura, largura, _ = img.shape

    integral = np.zeros(img.shape)
    resultado = np.zeros(img.shape)

    for y in range(altura):             #para cada linha y
        integral[y, 0] = img[y, 0]
        for x in range(1, largura):         #para cada coluna x, menos a primeira
            integral[y,x] = img[y,x] + integral[y-1, x]

    for y in range(1, altura):          #para cada linha y, menos a primeira      
        for x in range(largura):            #para cada linha y, menos a primeira   
            integral[y,x] = integral[y,x] + integral[y, x-1]   

    for y in range(margenVertical, altura-margenVertical):          #para cada linha y
        for x in range(margenHorizontal, largura-margenHorizontal):     #para cada coluna x
            soma = integral[y+margenVertical, x+margenHorizontal]               #canto inferior direito  
            soma = soma - integral[y-margenVertical-1, x+margenHorizontal]      #menos, canto superior direito  
            soma = soma - integral[y+margenVertical, x-margenHorizontal-1]      #menos, canto inferior esquerdo 
            soma = soma + integral[y-margenVertical-1, x-margenHorizontal-1]    #mais, canto superior esquerdo 
            resultado[y,x] = soma / (h*w)    

    return resultado

#===============================================================================

def main ():

    # Abre a imagem.
    img = cv2.imread (INPUT_IMAGE)

    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # colorida ou não. Também já convertemos para float32.
    img = img.astype (np.float32) / 255

    start_time = timeit.default_timer ()

    img1 = blurIngenuo (img, KSIZE)
    cv2.imshow ('01 - blurIngenuo', img1)
    cv2.imwrite ('01 - blurIngenuo.png', img1*255)
    if img1 is None:
        print ('Erro abrindo a imagem depois do blurIngenuo.\n')
        sys.exit () 

    img2 = blurSeparavel (img, KSIZE)
    cv2.imshow ('02 - blurSeparavel', img2)
    cv2.imwrite ('02 - blurSeparavel.png', img2*255)
    if img2 is None:
        print ('Erro abrindo a imagem depois do blurSeparavel.\n')
        sys.exit ()

    img3 = blurIntegrais (img, KSIZE)
    cv2.imshow ('03 - blurIntegrais', img3)
    cv2.imwrite ('03 - blurIntegrais.png', img3*255)
    if img3 is None:
        print ('Erro abrindo a imagem depois do blurIntegrais.\n')
        sys.exit ()
    
    print ('Tempo: %f' % (timeit.default_timer () - start_time))

    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
