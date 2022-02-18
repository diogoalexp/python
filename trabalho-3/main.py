#===============================================================================
# Trabalho 3
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#
# Nomes: Athena Andrômeda Stanislawski Gonçalves dos Santos, Celine Petterle Alberti,
#        Jéssica Dayane Ribeiro Gonçalves, Pietra Caroline de Pena
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'Wind Waker GC.bmp'
KSIZE = (17, 17)
SIGMA = 10
VEZES = 4

#===============================================================================


def brightPass (img):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(imgHLS, np.array([0,150,0]), np.array([255,255,255]))
    resultado = cv2.bitwise_and(img,img, mask= mask)

    return resultado

#-------------------------------------------------------------------------------
def bloomGaussiana (img, kSize, sigma, vezes):
    imgBrightPass = brightPass(img)

    imgBlur = cv2.GaussianBlur(imgBrightPass, (0,0), sigma)

    for x in range(vezes-1):
        sigma = sigma * 2
        imgBlur = cv2.GaussianBlur(imgBlur, (0,0), sigma)
  
    resultado = cv2.add(img, imgBlur)

    return resultado

#-------------------------------------------------------------------------------
def bloomBoxblur (img, kSize, vezes):
    h, w = kSize
    imgBrightPass = brightPass(img)

    imgBlur = cv2.blur(imgBrightPass, (h,w))

    for x in range(vezes-1):
        h = h*2
        w = w*2
        imgBlur = cv2.blur(imgBlur, (h,w))
  
    resultado = cv2.add(img, imgBlur)

    return resultado

#===============================================================================

def main ():

    # Abre a imagem.    
    img = cv2.imread (INPUT_IMAGE)


    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    start_time = timeit.default_timer ()

    img1 = bloomGaussiana (img, KSIZE, SIGMA, VEZES)
    cv2.imshow ('01 - BloomGaussiana', img1)
    cv2.imwrite ('01 - bloomGaussiana.png', img1)
    if img1 is None:
        print ('Erro abrindo a imagem depois do bloomGaussiana.\n')
        sys.exit () 

    img2 = bloomBoxblur (img, KSIZE, VEZES)
    cv2.imshow ('02 - bloomBoxblur', img2)
    cv2.imwrite ('02 - bloomBoxblur.png', img2)
    if img2 is None:
        print ('Erro abrindo a imagem depois do bloomBoxblur.\n')
        sys.exit ()
    
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
