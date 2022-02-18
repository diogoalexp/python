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

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 10

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!
    altura, largura, _ = img.shape
    for y in range(altura):             #para cada linha y
        for x in range(largura):            #para cada coluna x
            if img[y][x] > threshold:           #se I[y][x] > T
                img[y][x] = 1                       #I[y][x] ← objeto
            else:                               #senão
                img[y][x] = 0                       #I[y][x] ← fundo
    return img

#-------------------------------------------------------------------------------
def inunda (label, img, y0, x0, componente):

    img[y0][x0] = label
    altura, largura, _ = img.shape
    if(y0 + 1 < altura and img[y0+1][x0] == 1):
        inunda (label, img, y0+1, x0, componente) #vizinho de baixo

    if(x0+1 < largura and img[y0][x0+1] == 1):
        inunda (label, img, y0, x0+1, componente) #vizinho da direita

    if(x0 - 1 > 0 and img[y0][x0-1] == 1):
        inunda (label, img, y0, x0-1, componente) #vizinho da esquerda

    if(y0 - 1 > 0 and img[y0-1][x0] == 1):
        inunda (label, img, y0-1, x0, componente) #vizinho de cima
    
    if(componente['T'] > y0):
        componente['T'] = y0 #ponto mais alto

    if(componente['B'] < y0):
        componente['B'] = y0 #ponto mais baixo

    if(componente['L'] > x0):
        componente['L'] = x0 #ponto mais a esquerda

    if(componente['R'] < x0):
        componente['R'] = x0 #ponto mais a direita

    componente['n_pixels'] = componente['n_pixels'] + 1

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    list = []
    altura, largura, _ = img.shape
    label=2 
    for y in range(altura):
        for x in range(largura):
            if(img[y][x] == 1):
                componente = dict()
                componente['label'] = label  
                componente['n_pixels'] = 0 
                componente['T'] = y
                componente['L'] = x
                componente['B'] = y
                componente['R'] = x 
                inunda(label,img,y,x,componente)
                if( componente['n_pixels'] >= n_pixels_min):                    
                    if( (componente['R'] - componente['L']) >= largura_min):
                        if( (componente['B'] - componente['T']) >= altura_min):
                            list.append( componente )  
                label = label + 1

    return list
#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    if img is None:
        print ('Erro abrindo a imagem depois do binariza.\n')
        sys.exit ()

    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
