import numpy as np
from time import sleep as s

def sigmoid(z):
    return 1/(1+np.exp(-z))
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

refuerzo_adbitrario = .2

class ONN:
    def __init__(self,capas):
        self.capas = capas
        self.num_capas = len(capas)

        self.weights = [np.random.randn(y,x) for x,y in
                        zip(capas[:-1],capas[1:])]
        self.sesgos = [np.zeros((x,1)) for x in capas[1:]]

        self.umbrales = [np.random.random((x,1)) for x in capas[1:]]
        self.neuronas_que_superan_umbral = []
        self.error = float('inf')
    def forward_propagation(self,x):
        a = x
        activations = []
        #idx_weights_to_change = None
        for l in range(self.num_capas-1):
            z = np.dot(self.weights[l],a) + self.sesgos[l]
            a = tanh(z)
            #print('a',a)
            #print('u',self.umbrales[l])
            #print('a shape',a.shape,'umbrales shape',self.umbrales[l].shape)
            weights_to_change = a>self.umbrales[l]
            self.neuronas_que_superan_umbral.append(weights_to_change)
            activations.append(a)
        #print('neuronas a cambiar',self.neuronas_que_superan_umbral)
        return a

    def reinforce(self,respuesta_correcta,respuesta_emitida,minimo_esperado_para_reforzar=0.5,
                    n_neuronas_que_cumplen_requerimiento=1):
        expresion = respuesta_correcta-respuesta_emitida<minimo_esperado_para_reforzar
        n_neuronas_que_cumplen = len(expresion[expresion==True])
        magnitud_de_rf = refuerzo_adbitrario*n_neuronas_que_cumplen
        #print('n_neuronas_que_cumplen_requerimiento',n_neuronas_que_cumplen_requerimiento)
        #print('neuronas_que_cumplen',n_neuronas_que_cumplen)
        if np.all(expresion) or n_neuronas_que_cumplen>=n_neuronas_que_cumplen_requerimiento:
            #print('Sí')
            for c in range(self.num_capas-1):
                #print('self.weights[c]',self.weights[c])
                weights_a_cambiar = self.neuronas_que_superan_umbral[c].reshape(self.weights[c].shape[0],)
                #print('self.neuronas_que_superan_umbral[c]',self.neuronas_que_superan_umbral[c])
                #print('weights_a_cambiar',weights_a_cambiar)
                #print('antes',self.weights[c])
                self.weights[c][weights_a_cambiar,:] +=magnitud_de_rf
                #print('despues',self.weights[c])
                #s(20)
        else:
            for c in range(self.num_capas-1):
                weights_a_cambiar = self.neuronas_que_superan_umbral[c].reshape(self.weights[c].shape[0],)
                self.weights[c][weights_a_cambiar,:] +=magnitud_de_rf
        #    print('No')


#def difumination(self,ensayo):



def crear_estimulo(estimulo,inestabilidad=True):
    #estimulo = np.zeros((dims,1))
    if inestabilidad:
        estimulo = estimulo + np.random.uniform(-0.01,0.01,size=estimulo.shape)
    return estimulo

"""print('Con inestabilidad')

for i in range(5):
    print(crear_estimulo(e))
"""
"""print('Sin inestabilidad')
for i in range(5):
    print(crear_estimulo(e,False))
"""
e = np.array([[1],[1],[1]])
n_input_nodes = 3
n_output_nodes = 4
onn = ONN([n_input_nodes,5,n_output_nodes])
correct_response = np.array([[1],[1],[1],[1]])
ensayos = 0
cercania_entre_respuesta_emitida_y_esperada = 1.1
n_nodos_que_deden_cumplirse = 1
n_ensayos = 100
for i in range(n_ensayos):
    if ensayos%10==0:
        if cercania_entre_respuesta_emitida_y_esperada>.1:
            cercania_entre_respuesta_emitida_y_esperada -=.1
    if ensayos%8==0:
        if n_nodos_que_deden_cumplirse<n_output_nodes:
            n_nodos_que_deden_cumplirse +=1
    estimulo = crear_estimulo(e,False)
    emited_response = onn.forward_propagation(estimulo)
    #print('emited_response',emited_response)
    onn.reinforce(correct_response,emited_response,
                cercania_entre_respuesta_emitida_y_esperada,n_nodos_que_deden_cumplirse)
    ensayos +=1
    if i%(n_ensayos//10)==0:
        print('re',onn.forward_propagation(estimulo))
        print('cr',correct_response)
        print('Error: ',np.mean(np.abs(correct_response-emited_response)))
    #print('estimulo',estimulo)
    #print('emited_response',emited_response)
