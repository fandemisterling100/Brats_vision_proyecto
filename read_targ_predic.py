import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

with open('outs.pkl', 'rb') as f:
    data = pickle.load(f)

#imagen_original = data[0]
#imagen_original = target = imagen_original.cpu().numpy()
predicciones = data[1]
gt = data[0]

cont = 1

for i in range(0,len(predicciones)):

        segm = predicciones[i,:,:,0]
        anotacion = gt[i,:,:,0]

        id_segm = 'datos/' + str(cont) + '_prediccion.png'  
        id_gt = 'datos/' + str(cont) + '_anotacion.png'  

        matplotlib.image.imsave(id_segm, segm, cmap='gray')
        matplotlib.image.imsave(id_gt, anotacion, cmap='gray')
        cont +=1


    