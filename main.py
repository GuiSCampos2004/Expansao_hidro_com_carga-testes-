from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#####################################################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:31:00 2022

@author: alexmiranda
"""
import sympy as sp
from IPython.display import display_latex
from sympy.tensor.tensor import TensorIndexType, TensorHead, tensor_indices, tensor_heads, riemann_cyclic, TensorSymmetry
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, bsgs_direct_product, canonicalize, riemann_bsgs
#####################################################################################################################################
""""
Definições dos principais tensores que serão utilizados no código.
"""
d = sp.symbols('d', integer = True, positive = True)
a1, a2 = sp.symbols('a_1 a_2', integer = True, negative = False)
b1, b2 = sp.symbols('b_1 b_2', integer = True, positive = True)
Lorentz = TensorIndexType('Lorentz', dummy_name = 'pi', dim = d, metric_symmetry = 1, metric_name = 'g')
g = Lorentz.metric
u, grad, nabla, D = tensor_heads(r'u, \nabla, \nabla_\perp, \cal{D}', [Lorentz])
sigma = TensorHead(r'\sigma', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
Omega = TensorHead(r'\Omega', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
T = tensor_heads(r'T_{\;}', [])
mu = tensor_heads(r'\mu_{\;}', [])
Theta = tensor_heads(r'\Theta_{\;}', [])
Delta = TensorHead(r'\Delta', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
Ri = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
Fc = TensorHead(r'\cal{F}', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
Rc = TensorHead(r'\cal{R}', [Lorentz]*4, TensorSymmetry.riemann())
#####################################################################################################################################
""""
Introdução dos índices mudos (pi_0, pi_1, ..., pi_20) e dos índices livres gregos.
"""
pi_0, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9, pi_10, pi_11, pi_12, pi_13, pi_14, pi_15, pi_16, pi_17, pi_18,\
pi_19, pi_20 = tensor_indices('pi_0, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9, pi_10, pi_11, pi_12, pi_13, pi_14,\
pi_15, pi_16, pi_17, pi_18, pi_19, pi_20', Lorentz)
alpha, beta, gamma, delta, epsilon, zeta, eta, theta, vartheta, iota, kappa, landa, nu, xi, varpi, rho, tau, upsilon, phi,\
varphi, chi, psi, omega, varepsilon, varrho, varsigma, omicron = tensor_indices('alpha, beta, gamma, delta, epsilon, zeta, eta,\
theta, vartheta, iota, kappa, lambda, nu, xi, varpi, rho, tau, upsilon, phi, varphi, chi, psi, omega, varepsilon,\
varrho, varsigma, omicron', Lorentz)
Idx = [-alpha, -beta, -gamma, -delta, -epsilon, -zeta, -eta, -theta, -vartheta, -iota, -kappa, -landa, -nu, -xi, -varpi,\
-rho, -tau, -upsilon, -phi, -varphi, -chi, -psi, -omega, -varepsilon, -varrho, -varsigma, -omicron]
Idx_contra = [alpha, beta, gamma, delta, epsilon, zeta, eta, theta, vartheta, iota, kappa, landa, nu, xi, varpi, rho, tau,\
upsilon, phi, varphi, chi, psi, omega, varepsilon, varrho, varsigma, omicron]
Pi_contra = [pi_0, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9, pi_10, pi_11, pi_12, pi_13,\
pi_14, pi_15, pi_16, pi_17, pi_18, pi_19, pi_20]
Pi_covar = [-pi_0, -pi_1, -pi_2, -pi_3, -pi_4, -pi_5, -pi_6, -pi_7, -pi_8, -pi_9, -pi_10, -pi_11,\
-pi_12, -pi_13, -pi_14, -pi_15, -pi_16, -pi_17, -pi_18, -pi_19, -pi_20]
#####################################################################################################################################
Elem_fun_gz = [T(), mu(), u(Idx[0]), Ri(Idx[3], Idx[2], Idx[1], Idx[0])]
Elem_fun_dl = [T(), mu(), u(Idx[0]), Theta(), sigma(Idx[1], Idx[0]), Omega(Idx[1], Idx[0]), Ri(Idx[3], Idx[2], Idx[1], Idx[0])]
Elem_fun_cf = [u(Idx[0]), sigma(Idx[1], Idx[0]), Omega(Idx[1], Idx[0]), Fc(Idx[1], Idx[0]), Rc(Idx[3], Idx[2], Idx[1], Idx[0])]
#####################################################################################################################################
def avanca_indice(tensor):
    """
    Retorna o índice posterior na sequência da lista Idx ao primeiro índice do tensor de entrada.
    Exemplo: sigma(-beta, -alpha) = sigma(Idx[1], Idx[0]) retorna -gamma = Idx[2].
    """
    lista_indices = tensor.get_indices()
    ordem_dos_indices = []

    '''for i in range(len(lista_indices)):
        for j in range(len(Idx)):
            if lista_indices[i] == Idx[j] or lista_indices[i] == -Idx[j]:
                ordem_dos_indices.append(j)'''
                
    if len(lista_indices) == 0:
        indice = Idx[0]
    else:
        '''ind = ordem_dos_indices[0] + 1
        indice = Idx[ind]'''
        for j in range(len(Idx)):
            if lista_indices[0] == Idx[j] or lista_indices[0] == -Idx[j]:
                indice = Idx[j+1]

    return indice
#####################################################################################################################################
def troca_indices(tensor1, tensor2):
    """
    Troca os índices do tensor1 pela sequência de índices que começa no primeiro índice do tensor2.
    Exemplo: tensor1 = sigma(-beta, -alpha) = sigma(Idx[1], Idx[0]), tensor2 = Omega(-delta, -gamma) = omega(Idx[3], Idx[2])
    troca_indices(tensor1, tensor2) retorna o tensor sigma(-zeta, -epsilon) = sigma(Idx[5], Idx[4])
    """
    lista_indices1, lista_indices2 = tensor1.get_indices(), tensor2.get_indices()

    if len(lista_indices1) != 0 and len(lista_indices2) != 0:
        '''ordem_dos_indices1, ordem_dos_indices2 = [], []

        for i in range(len(lista_indices1)):
            for j in range(len(Idx)):
                if  lista_indices1[i] == Idx[j] or lista_indices1[i] == - Idx[j]:
                    ordem_dos_indices1.append(j) #armazena a ordem (na lista Idx) dos índices do tensor1

        for k in range(len(lista_indices2)):
            for l in range(len(Idx)):
                if lista_indices2[k] == Idx[l] or lista_indices2[k] == - Idx[l]:
                    ordem_dos_indices2.append(l) #armazena a ordem (na lista Idx) dos índices do tensor2'''

        for j in range(len(Idx)):
                if lista_indices2[0] == Idx[j] or lista_indices2[0] == -Idx[j]:
                    ordem_dos_indices2 = j #Guilherme: ordem_dos_indices2 agora tem informação apenas do primeiro indice

        tuplas_de_indices = []
        ind = ordem_dos_indices2
        for m in range(len(lista_indices1)):
            tuplas_de_indices.append((lista_indices1[m], Idx[ind + len(lista_indices1) - m]))
        tensor_novo = tensor1.substitute_indices(*tuplas_de_indices) #Guilherme: Checar se existe alguma outra forma de modificar os indices, se eu puder trocar os indices diretamente posso descartar a procura dos indices do tensor1 desta função
        #Guilherme Tentar implementar 'zip' aqui

    else:
        tensor_novo = tensor1

    return tensor_novo
#####################################################################################################################################
def inverte_indices(tensor):
        """
        Inverte os índices livres do tensor de entrada.
        Exemplo: tensor = sigma(-beta, -alpha)
             inverte_indices(tensor) retorna o tensor sigma(-alpha, -beta)
        """
        #antigo
        """lista_indices_completa = tensor.get_indices()
        comp_lista = len(lista_indices_completa) # tamanho da lista de índices do tensor de entrada
        lista_indices_invertida = lista_indices_completa[0:comp_lista]
        lista_indices_invertida.reverse()
        tupla_indices = []
        for i in range(comp_lista):
            tupla_indices.append((lista_indices_completa[i], lista_indices_invertida[i]))
        return tensor.substitute_indices(*tupla_indices)"""

        #novo
        #Maryana
        lista_indices_completa = tensor.get_indices()
        lista_indices_invertida = reversed(lista_indices_completa)
        tupla_indices = zip(lista_indices_completa, lista_indices_invertida)
        return tensor.substitute_indices(*tupla_indices)
#####################################################################################################################################
def compara_tensores(tensor1, tensor2):
    """
    Compara 2 tensores e resulta 'True' se os vetores são iguais ou proporcionais.
    """
    t1, t2 = tensor1, tensor2
    #display(t1, t2)
    if len(t1.get_indices()) != len(t2.get_indices()):
        resultado = False
    else:
        tupla_teste = (t2-t1).args[:]
        resultado = (len(tupla_teste) == 0 or tupla_teste[0].is_Add or tupla_teste[0].is_Integer)

    return resultado

def compara_tensor_lista(tensor, lista):
    """
    Compara o tensor de entrada com todos os tensores da lista de entrada.
    """
    resultado = False
    for i in range(len(lista)):
        if compara_tensores(tensor, lista[i]):
            resultado = True
    return resultado
#####################################################################################################################################
def reordena_lista(lista, inversao):
    """
    Reordena a lista de índices de entrada de modo que alpha esteja à esquerda de beta se inversao = 0 ou
    beta esteja à esquerda de alpha no caso de inversao = 1.
    """
    lista_nova = lista.copy()
    N = lista_nova.count(alpha) + lista_nova.count(beta)
    if N == 2:
        pos_alpha = lista_nova.index(alpha)
        pos_beta = lista_nova.index(beta)
        if (inversao == 0 and pos_alpha > pos_beta) or (inversao == 1 and pos_beta > pos_alpha):
            lista_nova[pos_alpha], lista_nova[pos_beta] = beta, alpha
    return lista_nova
#####################################################################################################################################
def ordem_normal_free(tensor, tipo_de_expansao): #mudado
    """
    Simplifica e ordena as partes de um tensor simbólico baseado na ordem dos elementos fundamentais
    e na quantidade de derivadas aplicadas. Retorna o tensor reorganizado e a ordem como lista de tuplas.
    """
    # Escolhe os elementos fundamentais conforme o tipo de expansão
    if tipo_de_expansao == 'dl': #Landau
        Elem_fund = Elem_fun_dl
    elif tipo_de_expansao == 'gz': #Grozdanov-Kapliz
        Elem_fund = Elem_fun_gz
    else: #Para fluidos conformes
        Elem_fund = Elem_fun_cf

    pos_R = len(Elem_fund) - 1 #Posição do tensor de Riemann (último)
    lista_partes_tensor = tensor.split() #quebra do tensor em suas diferentes partes, incluindo os nablas
    ord_nabla = 0 #número de nablas presentes num dado 'pedaço' do tensor, por exemplo, nabla(-beta)*u(-alpha)
    ord_normal = [] #lista que guarda informações das partes do tensor, incluindo o número de nablas em cada um

    for parte in lista_partes_tensor: #Varre cada "parte" do tensor: elemento fundamental + nablas
        Indices = parte.get_indices() #extrai índices do pedaço de tensor (ex: alpha, beta...)
        tuplas_de_indices = [(j, Idx[len(Indices) - 1 - i]) for i, j in enumerate(Indices)] #Armazena na tupla o índice junto com sua posição invertida
        tensor_parte_nova = parte.substitute_indices(*tuplas_de_indices) #Substitui o índice pelo Idx

        #Verifica se a parte do tensor é uma derivada
        if tensor_parte_nova == nabla(Idx[0]) or tensor_parte_nova == grad(Idx[0]) or tensor_parte_nova == D(Idx[0]):
            ord_nabla += 1
        else: #Caso não seja uma derivada, descobre qual elemento fundamental é
            for ord_elemf, elemf in enumerate(Elem_fund): #Varre os índices e valores de Elem_fund
                if tensor_parte_nova == elemf:
                    ord_normal.append((ord_elemf, ord_nabla)) #Armazena a ordem do elemento e ordem de sua derivada
                    ord_nabla = 0 #Zera o contador de derivadas

    n = len(ord_normal) #Salva número de elementos fundamentais encontrados

    # Ordena as tuplas: primeiro pelo tipo do elemento, depois pelo número de derivadas
    ord_normal.sort(key=lambda x: (x[0], x[1])) #Define uma função lambda (sem nome) que segue o critério de ordenação x[0], x[1]
    #MARYANA: A partir daqui está igual ao original, tentar deixar apenas uma estrutura de repetição
    lista_partes_nova = []

    for p in range(len(ord_normal)):
        lista_partes_nova.append(Elem_fund[ord_normal[p][0]])
        for q in range(ord_normal[p][1]): 
            if tipo_de_expansao == 'cf':
                lista_partes_nova[p] = D(avanca_indice(lista_partes_nova[p]))*lista_partes_nova[p]
            #elif (tipo_de_expansao == 'gz' or tipo_de_expansao == 'dl') and ord_normal[p][0] == pos_R: #Guilherme
            elif ord_normal[p][0] == pos_R:
                lista_partes_nova[p] = grad(avanca_indice(lista_partes_nova[p]))*lista_partes_nova[p]
            else:
                lista_partes_nova[p] = nabla(avanca_indice(lista_partes_nova[p]))*lista_partes_nova[p]

    tensor_ordem_normal = lista_partes_nova[n-1] 
    n_idx = len(lista_partes_nova[n-1].get_indices()) 

    for r in range(1, n):
        Indices = lista_partes_nova[n-(r+1)].get_indices()
        tuplas_de_indices = []
        ind = len(Indices) - 1
        for s in range(len(Indices)):
            tuplas_de_indices.insert(0, (Indices[ind-s], Idx[n_idx+s]))

        n_idx = n_idx + len(Indices)
        tensor_substituido = lista_partes_nova[n-(r+1)].substitute_indices(*tuplas_de_indices)
        tensor_ordem_normal = tensor_substituido*tensor_ordem_normal

    #display(tensor_ordem_normal, ord_normal) #Não está no original! Apenas printa tensores para ver comportamento da função.


    return tensor_ordem_normal, ord_normal #Retorna o tensor reordenado e reconstruído e a sua ordem de elemento e derivadas
#####################################################################################################################################
def derivada_ordem_normal(ordem_normal, tipo_de_expansao):
    """
    Produz o tensor resultante da derivação do tensor associado à lista ordem_normal de entrada.
    """
    tipo_exp = tipo_de_expansao
    if tipo_exp == 'dl':
        Elem_fund = Elem_fun_dl
    elif tipo_exp == 'gz':
        Elem_fund = Elem_fun_gz
    else:
        Elem_fund = Elem_fun_cf

    n_elem = len(Elem_fund)
    pos_R = n_elem - 1
    n = len(ordem_normal)
    ord_normal = []

    for i in range(n):
        ord_normal.append(list(ordem_normal[i]))

    derivada_tensor = []

    for j in range(n):
        ord_normal[j][1] = ord_normal[j][1] + 1
        lista_partes_nova = []

        for k in range(n):
            lista_partes_nova.append(Elem_fund[ord_normal[k][0]])
            for l in range(ord_normal[k][1]):
                if tipo_exp == 'cf':
                    lista_partes_nova[k] = D(avanca_indice(lista_partes_nova[k]))*lista_partes_nova[k]
                #elif (tipo_exp == 'gz' or tipo_exp == 'dl') and ord_normal[k][0] == pos_R: #Guilherme
                elif ord_normal[k][0] == pos_R:
                     lista_partes_nova[k] = grad(avanca_indice(lista_partes_nova[k]))*lista_partes_nova[k]
                else:
                    lista_partes_nova[k] = nabla(avanca_indice(lista_partes_nova[k]))*lista_partes_nova[k]

        ord_normal[j][1] = ord_normal[j][1] - 1
        elemento_ordem_normal = lista_partes_nova[n-1]
        n_idx = len(lista_partes_nova[n-1].get_indices())

        for r in range(1, n):
            Indices = lista_partes_nova[n-(r+1)].get_indices()
            tuplas_de_indices = []
            ind = len(Indices) - 1
            for s in range(len(Indices)):
                tuplas_de_indices.insert(0, (Indices[ind-s], Idx[n_idx+s]))

            n_idx = n_idx + len(Indices)
            tensor_substituido = lista_partes_nova[n-(r+1)].substitute_indices(*tuplas_de_indices)
            elemento_ordem_normal = tensor_substituido*elemento_ordem_normal

        novo_tensor_ord_normal = ordem_normal_free(elemento_ordem_normal, tipo_exp)[0]
        #Guilherme: Verificar qual a entrada inserida nessa função, se a entrada já estiver arrumada a linha acima não tem sentido
        derivada_tensor.append(novo_tensor_ord_normal)

    return derivada_tensor
#####################################################################################################################################
def ingredientes(ordem_hidro, curvatura, tipo_de_expansao):
    """
    Construção dos ingredientes para a expansão hidrodinâmica relativística desde a 0-ésima até a n-ésima ordem.
    """
    tipo_exp = tipo_de_expansao
    if tipo_exp == 'dl':
        I0 = [u(Idx[0]), g(Idx[1], Idx[0])]
        I1 = [Theta(), nabla(Idx[0])*T(), nabla(Idx[0])*mu(), sigma(Idx[1], Idx[0]), Omega(Idx[1], Idx[0])]
        M0, M1 = [1, 2], [0, 1, 1, 2, 2]
        n_elem = 7

    elif tipo_exp == 'gz':
        I0, I1 = [u(Idx[0]), g(Idx[1], Idx[0])], [nabla(Idx[0])*T(), nabla(Idx[0])*mu(), nabla(Idx[1])*u(Idx[0])]
        M0, M1 = [1, 2], [1, 1, 2]
        n_elem = 4

    else:
        I0 = [u(Idx[0]), g(Idx[1], Idx[0])]
        I1 = [sigma(Idx[1], Idx[0]), Omega(Idx[1], Idx[0])]
        M0, M1 = [1, 2], [2, 2]
        n_elem = 5

    pos_R = n_elem - 1
    lista_ing, lista_M  = [I0], [M0]


    '''if ordem_hidro != 0:
        lista_ing.append(I1)
        lista_M.append(M1)
        if ordem_hidro == 1:
            lista_ing_final = lista_ing'''

    #Guilherme: O código é fechado, não há por que colocar uma ordem_hidro menor que 1, então não é necessário intervir no caso ordem_hidro == 0
    if ordem_hidro == 1:
          lista_ing_final = lista_ing
    else:
        lista_ing.append(I1)
        lista_M.append(M1)

    for n in range(2, ordem_hidro + 1): #esse loop varia sobre as ordens da hidrodinâmica, desde a segunda até a n-ésima ordem.
        I_temp = []
        for i in range(len(lista_ing[n-1])): #aqui são contruídos os elementos obtidos das derivadas da ordem anterior
            lista_ordem_normal = ordem_normal_free(lista_ing[n-1][i], tipo_exp)[1]
            temp1 = derivada_ordem_normal(lista_ordem_normal, tipo_exp)

            for j in range(len(temp1)):
                if compara_tensor_lista(temp1[j], I_temp) == False:
                    I_temp.append(temp1[j])

        limite = n//2
        for b2 in range(1, limite + 1): #aqui são contruídos os elementos obtidos dos produtos de elementos de ordens anteriores
            B1 = sp.solve(b1 + b2 - n, b1)
            for k in range(len(lista_ing[B1[0]])):
                for l in range(len(lista_ing[b2])):
                    tensor_temp = troca_indices(lista_ing[B1[0]][k], lista_ing[b2][l])
                    temp2 = ordem_normal_free(tensor_temp*lista_ing[b2][l], tipo_exp)[0]
                    if compara_tensor_lista(temp2, I_temp) == False:
                        I_temp.append(temp2)

        #testa se o tensor de Riemann e/ou o tensor F devem ser incluídos na lista de ingredientes
        '''if (tipo_exp == 'gz' or tipo_exp == 'dl') and n == 2 and curvatura == 1:
            I_temp.append(Ri(Idx[3], Idx[2], Idx[1], Idx[0]))
        if tipo_exp == 'cf' and n == 2 and curvatura == 1:
            I_temp.append(Fc(Idx[1], Idx[0]))
            I_temp.append(Rc(Idx[3], Idx[2], Idx[1], Idx[0]))'''

        #Guilherme: Não tenho certeza, mas acho que esse arranjo exige menos processamento
        if n == 2 and curvatura == 1:
            if(tipo_exp == 'gz' or tipo_exp == 'dl'):
                I_temp.append(Ri(Idx[3], Idx[2], Idx[1], Idx[0]))

            else:
                I_temp.append(Fc(Idx[1], Idx[0]))
                I_temp.append(Rc(Idx[3], Idx[2], Idx[1], Idx[0]))
        

        lista_ing.append(I_temp)
        lista_ing_final = [lista_ing[0]]

        for i in range(1, len(lista_ing)): #realiza a mudança da ordem dos elementos, colocando os Riemann's na parte final da lista
            lista_ing_fluido, lista_ing_curvatura, lista_ing_temp = [], [], []

            for j in range(len(lista_ing[i])):
                ord_tens_temp = ordem_normal_free(lista_ing[i][j], tipo_exp)[1]

                for k in range(len(ord_tens_temp)):
                    if ord_tens_temp[k][0] == pos_R:
                        indica_curvatura = True
                    else:
                        indica_curvatura = False

                if indica_curvatura:
                    lista_ing_curvatura.append(lista_ing[i][j])
                else:
                    lista_ing_fluido.append(lista_ing[i][j])

            for l in range(len(lista_ing_fluido)):
                lista_ing_temp.append(lista_ing_fluido[l])

            for p in range(len(lista_ing_curvatura)):
                lista_ing_temp.append(lista_ing_curvatura[p])

            lista_ing_final.append(lista_ing_temp)

        M = []

        for m in range(len(lista_ing_final[n])):
            num = len(lista_ing_final[n][m].get_indices())
            M.append(num)

        lista_M.append(M)

    return lista_ing_final, lista_M
#####################################################################################################################################
def estruturas_tensoriais(ordem_hidro, grau_tens, ingredientes, Matriz_M, tipo_de_expansao):
    """
    Obtenção das estruturas tensoriais possíveis para um dado grau tensorial: N = 0 (escalar), N= 1 (vetor)
    e N = 2 (tensor). Observe que são armazenadas as estruturas reduzidas, uma vez que os elementos envolvendo
    a métrica foram retirados, pois na sequência serão realizadas todas as contrações possíveis, e os elementos
    em 'u' que restaram são aqueles que multiplicam o tensor de Riemann.
    """
    tipo_exp = tipo_de_expansao
    if tipo_exp == 'gz':
        n_elem = 4
    elif tipo_exp == 'dl':
        n_elem = 7
    else:
        n_elem = 5
    pos_R = n_elem - 1
    N = grau_tens
    I, I0 = ingredientes[ordem_hidro], ingredientes[0]
    M = Matriz_M[ordem_hidro]
    Lista_Est_red = []
    for m in range(1, len(M)+1): # Esse 2º laço varre os valores possíveis de m, ou seja, sobre todos os ingredientes I.
        J, J_red, Estruturas_red = [], [], []
        final = int((M[m-1]+N+2)/2)
        for a2 in range(0, final):
            A1_plus = sp.solve(a1+2*a2-M[m-1]-N, a1)
            A1 = A1_plus
            if N != 0:
                A1_minus = sp.solve(a1+2*a2-M[m-1]+N, a1)
                A1 = A1 + A1_minus
            for i in range(len(A1)): #Esse 5º laço varre a lista A1 para adicionar as soluções (a1,a2) à J.
                indica_presenca = False
                for j in range(len(J)):
                    if (A1[i], a2) == J[j]:
                        indica_presenca = True
                if not indica_presenca: #Uma solução (a1, a2) só é adicionado se ela não estiver presente em J.
                    est1, est2 = 1, 1
                    for k in range(a2):
                        ind1 = M[m-1] + A1[i] + 2*k + 1
                        ind2 = M[m-1] + A1[i] + 2*k
                        est1 = I0[1].substitute_indices((Idx[1], Idx[ind1]), (Idx[0], Idx[ind2]))*est1
                    for l in range(A1[i]):
                        ind = M[m-1] + l
                        est2 = I0[0].substitute_indices((Idx[0], Idx[ind]))*est2
                    J.append((A1[i], a2))
                    I_decomposto = I[m-1].split()
                    if pos_R % 3 == 0 and (A1[i] == 0 or I_decomposto[len(I_decomposto) - 1] == Ri(Idx[3], Idx[2], Idx[1], Idx[0])):
                        J_red.append((A1[i], a2))
                        Est_tens = est2*I[m-1]
                        Estruturas_red.append(Est_tens)
                        if not compara_tensor_lista(Est_tens, Lista_Est_red):
                            Lista_Est_red.append(Est_tens)
                    elif pos_R % 3 != 0 and (A1[i] == 0 or I_decomposto[len(I_decomposto) - 1] == Fc(Idx[1], Idx[0])\
                                              or I_decomposto[len(I_decomposto) - 1] == Rc(Idx[3], Idx[2], Idx[1], Idx[0])):
                        J_red.append((A1[i], a2))
                        Est_tens = est2*I[m-1]
                        Estruturas_red.append(Est_tens)
                        if not compara_tensor_lista(Est_tens, Lista_Est_red):
                            Lista_Est_red.append(Est_tens)
    return Lista_Est_red
#####################################################################################################################################
def constroi_BSGS(ordem_normal, tipo_de_expansao):
    """
    Construção da base e do conjunto forte de geradores (gens) para o grupo de simetrias do tensor associado
    à lista 'ordem_normal'. A base e o gens são utilizados posteriormente no processo de canonicalização do tensor.
    """
    tipo_exp = tipo_de_expansao
    indica_troca_tensor = True #variável que indica a troca de uma 'parte tensorial' para outra, incluídos os nablas.
    count = 1
    lista_canonica = []
    for k in range(len(ordem_normal)):
        if k != 0:
            if ordem_normal[k] == ordem_normal[k-1]:
                count = count + 1
                indica_troca_tensor = False
            else:
                tensor_temp = (base, gens, count, 0)
                lista_canonica.append(tensor_temp)
                indica_troca_tensor = True
                count = 1
        if indica_troca_tensor:
            num_nabla = ordem_normal[k][1]
            base1, gens1 = get_symmetric_group_sgs(1)
            base_sigma, gens_sigma = get_symmetric_group_sgs(2, antisym = False)
            base_omega, gens_omega = get_symmetric_group_sgs(2, antisym = True)
            base_nabla, gens_nabla = get_symmetric_group_sgs(num_nabla)
            if tipo_exp == 'gz':
                if ordem_normal[k][0] == 0 or ordem_normal[k][0] == 1:
                    base, gens = base_nabla, gens_nabla
                elif ordem_normal[k][0] == 2:
                    if ordem_normal[k][1] == 0:
                        base, gens = base1, gens1
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, base1, gens1)
                elif ordem_normal[k][0] == 3:
                    if ordem_normal[k][1] == 0:
                        base, gens =  riemann_bsgs[0], riemann_bsgs[1]
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, riemann_bsgs[0], riemann_bsgs[1])
            elif tipo_exp == 'dl':
                if ordem_normal[k][0] == 0 or ordem_normal[k][0] == 1 or ordem_normal[k][0] == 3:
                    base, gens = base_nabla, gens_nabla
                elif ordem_normal[k][0] == 2:
                    if ordem_normal[k][1] == 0:
                        base, gens = base1, gens1
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, base1, gens1)
                elif ordem_normal[k][0] == 4:
                    if ordem_normal[k][1] == 0:
                        base, gens = base_sigma, gens_sigma
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, base_sigma, gens_sigma)
                elif ordem_normal[k][0] == 5:
                    if ordem_normal[k][1] == 0:
                        base, gens = base_omega, gens_omega
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, base_omega, gens_omega)
                elif ordem_normal[k][0] == 6:
                    if ordem_normal[k][1] == 0:
                        base, gens =  riemann_bsgs[0], riemann_bsgs[1]
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, riemann_bsgs[0], riemann_bsgs[1])
            else:
                if ordem_normal[k][0] == 0:
                    if ordem_normal[k][1] == 0:
                        base, gens = base1, gens1
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, base1, gens1)
                elif ordem_normal[k][0] == 1:
                    if ordem_normal[k][1] == 0:
                        base, gens = base_sigma, gens_sigma
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, base_sigma, gens_sigma)
                elif ordem_normal[k][0] == 2 or ordem_normal[k][0] == 3:
                    if ordem_normal[k][1] == 0:
                        base, gens = base_omega, gens_omega
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, base_omega, gens_omega)
                elif ordem_normal[k][0] == 4:
                    if ordem_normal[k][1] == 0:
                        base, gens =  riemann_bsgs[0], riemann_bsgs[1]
                    else:
                        base, gens = bsgs_direct_product(base_nabla, gens_nabla, riemann_bsgs[0], riemann_bsgs[1])
    tensor_temp = (base, gens, count, 0)
    lista_canonica.append(tensor_temp)
    return lista_canonica
#####################################################################################################################################
def constroi_BSGS_tensorial(ordem_normal_idx, tipo_de_expansao):
    """
    Construção da base e do conjunto forte de geradores (gens) para o grupo de simetrias do tensor associado
    à lista 'ordem_normal_idx', que é complementada pelos índices mudos e livres que formam o tensor. Esse método
    é utilizado para implementar uma simetria entre os 'slots' ocupados pelos índices alpha e beta, uma vez que
    as estruturas tensoriais devem ser TST, ou seja, simétricas, transversais e livres de traço.
    """
    tipo_exp = tipo_de_expansao
    ord_modificada = []
    for i in range(len(ordem_normal_idx)):
        lista_temp = list(ordem_normal_idx[i])
        for j in range(2, len(lista_temp)):
            if Idx_contra.count(lista_temp[j]) != 0:
                lista_temp[j] = alpha
            else:
                lista_temp[j] = pi_0
        ord_modificada.append(lista_temp)
    indica_troca_tensor = True
    count = 1
    lista_canonica = []
    for k in range(len(ord_modificada)):
        if k != 0:
            t0_rank, n0_dif = ord_modificada[k-1][0], ord_modificada[k-1][1]
            t1_rank, n1_dif = ord_modificada[k][0], ord_modificada[k][1]
            num0_latin, num1_latin = ord_modificada[k-1].count(alpha), ord_modificada[k].count(alpha)
            if (t0_rank, n0_dif) == (t1_rank, n1_dif) and num0_latin == num1_latin:
                count = count + 1
                indica_troca_tensor = False
            else:
                tensor_temp = (base, gens, count, 0)
                lista_canonica.append(tensor_temp)
                indica_troca_tensor = True
                count = 1
        if indica_troca_tensor:
            num_nabla = ord_modificada[k][1]
            base1, gens1 = get_symmetric_group_sgs(1)
            base_sigma, gens_sigma = get_symmetric_group_sgs(2, antisym = False)
            base_omega, gens_omega = get_symmetric_group_sgs(2, antisym = True)
            base_nabla_mod, gens_nabla_mod = get_symmetric_group_sgs(num_nabla)
            #base_nabla_mod, gens_nabla_mod = [], [Permutation(num_nabla + 1)]
            if tipo_exp == 'gz':
                if ord_modificada[k][0] == 0 or ord_modificada[k][0] == 1:
                    base, gens = base_nabla_mod, gens_nabla_mod
                elif ord_modificada[k][0] == 2:
                    if ord_modificada[k][1] == 0:
                        base, gens = base1, gens1
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, base1, gens1)
                elif ord_modificada[k][0] == 3:
                    if ord_modificada[k][1] == 0:
                        base, gens =  riemann_bsgs[0], riemann_bsgs[1]
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, riemann_bsgs[0], riemann_bsgs[1])
            elif tipo_exp == 'dl':
                if ord_modificada[k][0] == 0 or ord_modificada[k][0] == 1 or ord_modificada[k][0] == 3:
                    base, gens = base_nabla_mod, gens_nabla_mod
                elif ord_modificada[k][0] == 2:
                    if ord_modificada[k][1] == 0:
                        base, gens = base1, gens1
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, base1, gens1)
                elif ord_modificada[k][0] == 4:
                    if ord_modificada[k][1] == 0:
                        base, gens = base_sigma, gens_sigma
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, base_sigma, gens_sigma)
                elif ord_modificada[k][0] == 5:
                    if ord_modificada[k][1] == 0:
                        base, gens = base_omega, gens_omega
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, base_omega, gens_omega)
                elif ord_modificada[k][0] == 6:
                    if ord_modificada[k][1] == 0:
                        base, gens = riemann_bsgs[0], riemann_bsgs[1]
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, riemann_bsgs[0], riemann_bsgs[1])
            else:
                if ord_modificada[k][0] == 0:
                    if ord_modificada[k][1] == 0:
                        base, gens = base1, gens1
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, base1, gens1)
                elif ord_modificada[k][0] == 1:
                    if ord_modificada[k][1] == 0:
                        base, gens = base_sigma, gens_sigma
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, base_sigma, gens_sigma)
                elif ord_modificada[k][0] == 2 or ord_modificada[k][0] == 3:
                    if ord_modificada[k][1] == 0:
                        base, gens = base_omega, gens_omega
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, base_omega, gens_omega)
                elif ord_modificada[k][0] == 4:
                    if ord_modificada[k][1] == 0:
                        base, gens = riemann_bsgs[0], riemann_bsgs[1]
                    else:
                        base, gens = bsgs_direct_product(base_nabla_mod, gens_nabla_mod, riemann_bsgs[0], riemann_bsgs[1])
    tensor_temp = (base, gens, count, 0)
    lista_canonica.append(tensor_temp)
    return lista_canonica
#####################################################################################################################################
def tensor_forma_canonica(grau_tens, lista_numerica, ordem_normal, simplificador, tipo_de_expansao):
    """
    Canonicalização do tensor associado à ordem normal. A lista numérica guarda as posições dos índices
    de acordo com a sequência (alpha, beta, pi_0, -pi_0, pi_1, -pi_1,..., pi_n, -pi_n). A ordem_normal é
    somente numérica no caso de simplificador = False, ao passo que inclui os índices no caso de
    simplificador = True. Esse último caso é implementado como uma simplificação adicional das estruturas,
    que leva em conta a simetria entre alpha e beta no caso de estruturas tensoriais de ordem 2.
    """
    tipo_exp = tipo_de_expansao
    N = grau_tens
    lista_nova = lista_numerica.copy()
    nova_ordem_normal = ordem_normal.copy()
    n_idx = len(lista_numerica)
    for i in range(len(ordem_normal)):
        if tipo_exp == 'dl' and ordem_normal[i] == (3, 0): #remove os elementos associados ao escalar Theta
            nova_ordem_normal.remove((3, 0))
    lista_nova.append(len(lista_numerica))
    lista_nova.append(len(lista_numerica) + 1)
    G = Permutation(lista_nova)
    if simplificador:
        lista_canonica = constroi_BSGS_tensorial(nova_ordem_normal, tipo_exp)
        dummies = [range(0, N), range(N, n_idx)]
        lista_zeros = [0, 0]
        ordem_canonica = canonicalize(G, dummies, lista_zeros, *lista_canonica)
    else:
        lista_canonica = constroi_BSGS(nova_ordem_normal, tipo_exp)
        ordem_canonica = canonicalize(G, range(N, n_idx), 0, *lista_canonica)
    if isinstance(ordem_canonica, list):
        lista_final = []
        len_ord_red = len(ordem_canonica) - 2
        for j in range(len_ord_red):
            if (N == 2 or N == 1) and ordem_canonica[j] == 0:
                lista_final.append(alpha)
            elif N == 2 and ordem_canonica[j] == 1:
                lista_final.append(beta)
            elif (ordem_canonica[j] - N) % 2 == 0: #armazenam-se aqui os índices numa sequência inversa, de pi_n à pi_0
                lista_final.append(Pi_contra[int((len_ord_red-N)/2) - int((ordem_canonica[j]-N)/2) - 1])
            else:
                lista_final.append(Pi_covar[int((len_ord_red-N)/2) - int((ordem_canonica[j]-N)/2) - 1])
    else:
        lista_final = ordem_canonica
    return lista_final
#####################################################################################################################################
def ordem_com_indices(tensor, tipo_de_expansao):
    """
    Obtenção da ordem normal com índices associada ao tensor de entrada.
    """
    tipo_exp = tipo_de_expansao
    lista_partes_tensor = tensor.split() #quebra do tensor em suas diferentes partes, incluindo os nablas.
    ord_nabla = 0 #número de nablas presentes num dado 'pedaço' do tensor, por exemplo, nabla(a)*u(b).
    ord_normal = [] #lista que guarda info dos 'pedaços' do tensor, incluindo o número de nablas em cada um.
    Indices_do_tensor, lista_idx_mudo, lista_pares = [], [], []
    n_idx, conta_mudo = 0, 0
    for i in range(len(lista_partes_tensor)):
        Indices = lista_partes_tensor[i].get_indices()
        n_idx = n_idx + len(lista_partes_tensor[i].get_indices())
        lista_pares_interno = []
        for j in range(len(Indices)):
            ind_mudo = []
            if Idx.count(Indices[j]) != 0 or Idx_contra.count(Indices[j]) != 0:
                Indices_do_tensor.append(Indices[j])
            else:
                if Pi_contra.count(Indices[j]) != 0:
                    ind_mudo.append(Pi_contra.index(Indices[j]))
                    ind_mudo.append(1)
                else:
                    ind_mudo.append(Pi_covar.index(Indices[j]))
                    ind_mudo.append(-1)
                if lista_idx_mudo.count(ind_mudo) == 0:
                    if ind_mudo[1] == 1 or len(lista_pares_interno) == 0:
                        lista_idx_mudo.append(ind_mudo)
                        Indices_do_tensor.append(Indices[j])
                        if ind_mudo[1] == 1:
                            conta_mudo = conta_mudo + 1
                    else:
                        encontra_par = False
                        for k in range(len(lista_pares_interno)):
                            if not encontra_par and ind_mudo[0] == lista_pares_interno[k][1]:
                                ind_mudo[0] = lista_pares_interno[k][0]
                                novo_indice = Pi_covar[ind_mudo[0]]
                                encontra_par = True
                                lista_idx_mudo.append(ind_mudo)
                                Indices_do_tensor.append(novo_indice)
                            else:
                                lista_idx_mudo.append(ind_mudo)
                                Indices_do_tensor.append(Indices[j])
                else:
                    if ind_mudo[1] == 1:
                        temp_new = ind_mudo[0]
                        ind_mudo[0] = conta_mudo
                        lista_pares.append((conta_mudo, temp_new)) #armazena a posição do novo e do velho índice mudo na lista dos Pi's
                        lista_pares_interno.append((conta_mudo, temp_new))
                        novo_indice = Pi_contra[ind_mudo[0]]
                    else:
                        encontra_par = False
                        for k in range(len(lista_pares)):
                            if not encontra_par and ind_mudo[0] == lista_pares[k][1]:
                                ind_mudo[0] = lista_pares[k][0]
                                novo_indice = Pi_covar[ind_mudo[0]]
                                encontra_par = True
                    lista_idx_mudo.append(ind_mudo)
                    Indices_do_tensor.append(novo_indice)
                    if ind_mudo[1] == 1:
                        conta_mudo = conta_mudo + 1
        T_parte_nova = lista_partes_tensor[i]
        if len(Indices) == 1:
            if T_parte_nova == nabla(Indices[0]) or T_parte_nova == grad(Indices[0]) or T_parte_nova == D(Indices[0]):
                ord_nabla = ord_nabla + 1
            elif T_parte_nova == u(Indices[0]):
                if tipo_exp == 'gz' or tipo_exp == 'dl':
                    ord_elem = 2
                else:
                    ord_elem = 0
                ord_normal.append((ord_elem, ord_nabla,*Indices_do_tensor))
                Indices_do_tensor = []
                ord_nabla = 0
        else:
            if tipo_exp == 'gz':
                if len(Indices) == 0:
                    if T_parte_nova == T():
                        ord_elem = 0
                    elif T_parte_nova == mu():
                        ord_elem = 1
                elif len(Indices) == 4:
                    ord_elem = 3
            elif tipo_exp == 'dl':
                if len(Indices) == 0:
                    if T_parte_nova == T():
                        ord_elem = 0
                    elif T_parte_nova == mu():
                        ord_elem = 1
                    elif T_parte_nova == Theta():
                        ord_elem = 3
                elif len(Indices) == 2:
                    if T_parte_nova == sigma(Indices[0], Indices[1]):
                        ord_elem = 4
                    elif T_parte_nova == Omega(Indices[0], Indices[1]):
                        ord_elem = 5
                elif len(Indices) == 4:
                    ord_elem = 6
            elif tipo_exp == 'cf':
                if len(Indices) == 2:
                    if T_parte_nova == sigma(Indices[0], Indices[1]):
                        ord_elem = 1
                    elif T_parte_nova == Omega(Indices[0], Indices[1]):
                        ord_elem = 2
                    elif T_parte_nova == Fc(Indices[0], Indices[1]):
                        ord_elem = 3
                elif len(Indices) == 4:
                    ord_elem = 4
            ord_normal.append((ord_elem, ord_nabla,*Indices_do_tensor))
            Indices_do_tensor = []
            ord_nabla = 0
    return ord_normal
#####################################################################################################################################
def reconstroi_tensor(ord_normal, tipo_de_expansao):
    """
    Reconstrução de tensor a partir da entrada da ordem normal com índices.
    """
    tipo_exp = tipo_de_expansao
    count_ind = 0
    T_temp = 1
    tupla_de_indices = []
    for p in range(len(ord_normal)-1, -1, -1):
        num1 = len(ord_normal[p])
        if tipo_exp == 'gz':
            if ord_normal[p][0] == 0:
                Elem_da_lista = T()
            elif ord_normal[p][0] == 1:
                Elem_da_lista = mu()
            elif ord_normal[p][0] == 2:
                Elem_da_lista = u(Idx[count_ind])
                tupla_de_indices.insert(0, (Idx[count_ind], ord_normal[p][num1-1]))
                count_ind = count_ind + 1
            elif ord_normal[p][0] == 3:
                n2_dif = ord_normal[p][1]
                for q in range(5, 1, -1):
                    tupla_de_indices.insert(0, (Idx[5-q+count_ind], ord_normal[p][q+n2_dif]))
                Elem_da_lista = Ri(Idx[count_ind+3], Idx[count_ind+2], Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 4
        elif tipo_exp == 'dl':
            if ord_normal[p][0] == 0:
                Elem_da_lista = T()
            elif ord_normal[p][0] == 1:
                Elem_da_lista = mu()
            elif ord_normal[p][0] == 2:
                Elem_da_lista = u(Idx[count_ind])
                tupla_de_indices.insert(0, (Idx[count_ind], ord_normal[p][num1-1]))
                count_ind = count_ind + 1
            elif ord_normal[p][0] == 3:
                Elem_da_lista = Theta()
            elif ord_normal[p][0] == 4:
                n3_dif = ord_normal[p][1]
                for q in range(3, 1, -1):
                    tupla_de_indices.insert(0, (Idx[3-q+count_ind], ord_normal[p][q+n3_dif]))
                Elem_da_lista = sigma(Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 2
            elif ord_normal[p][0] == 5:
                n4_dif = ord_normal[p][1]
                for r in range(3, 1, -1):
                    tupla_de_indices.insert(0, (Idx[3-r+count_ind], ord_normal[p][r+n4_dif]))
                Elem_da_lista = Omega(Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 2
            else:
                n5_dif = ord_normal[p][1]
                for s in range(5, 1, -1):
                    tupla_de_indices.insert(0, (Idx[5-s+count_ind], ord_normal[p][s+n5_dif]))
                Elem_da_lista = Ri(Idx[count_ind+3], Idx[count_ind+2], Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 4
        else:
            if ord_normal[p][0] == 0:
                Elem_da_lista = u(Idx[count_ind])
                tupla_de_indices.insert(0, (Idx[count_ind], ord_normal[p][num1-1]))
                count_ind = count_ind + 1
            elif ord_normal[p][0] == 1:
                n1_dif = ord_normal[p][1]
                for q in range(3, 1, -1):
                    tupla_de_indices.insert(0, (Idx[3-q+count_ind], ord_normal[p][q+n1_dif]))
                Elem_da_lista = sigma(Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 2
            elif ord_normal[p][0] == 2:
                n2_dif = ord_normal[p][1]
                for r in range(3, 1, -1):
                    tupla_de_indices.insert(0, (Idx[3-r+count_ind], ord_normal[p][r+n2_dif]))
                Elem_da_lista = Omega(Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 2
            elif ord_normal[p][0] == 3:
                n3_dif = ord_normal[p][1]
                for r in range(3, 1, -1):
                    tupla_de_indices.insert(0, (Idx[3-r+count_ind], ord_normal[p][r+n3_dif]))
                Elem_da_lista = Fc(Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 2
            else:
                n4_dif = ord_normal[p][1]
                for s in range(5, 1, -1):
                    tupla_de_indices.insert(0, (Idx[5-s+count_ind], ord_normal[p][s+n4_dif]))
                Elem_da_lista = Rc(Idx[count_ind+3], Idx[count_ind+2], Idx[count_ind+1], Idx[count_ind])
                count_ind = count_ind + 4
        for m in range(ord_normal[p][1]-1, -1, -1):
            if tipo_exp == 'cf':
                Elem_da_lista = D(Idx[count_ind])*Elem_da_lista
                tupla_de_indices.insert(0, (Idx[count_ind], ord_normal[p][2+m]))
                count_ind = count_ind + 1
            elif (tipo_exp == 'gz' and ord_normal[p][0] == 3) or (tipo_exp == 'dl' and ord_normal[p][0] == 6):
                Elem_da_lista = grad(Idx[count_ind])*Elem_da_lista
                tupla_de_indices.insert(0, (Idx[count_ind], ord_normal[p][2+m]))
                count_ind = count_ind + 1
            else:
                Elem_da_lista = nabla(Idx[count_ind])*Elem_da_lista
                tupla_de_indices.insert(0, (Idx[count_ind], ord_normal[p][2+m]))
                count_ind = count_ind + 1
        T_temp = Elem_da_lista*T_temp
    tensor_ordem_normal = T_temp.substitute_indices(*tupla_de_indices)
    return tensor_ordem_normal
#####################################################################################################################################
def reordena_indices(ordem_normal_reduzida):
    """"
    Recoloca os índices na forma canônica, de acordo com a sequência (alpha, beta, pi_0, -pi_0, pi_1, -pi_1,..., pi_n, -pi_n).
    """
    n = len(ordem_normal_reduzida)
    ord_dummy_red = ordem_normal_reduzida.copy()
    for r in range(n, 0, -1): #Este é o algoritmo da bolha, que irá colocar as tuplas em ordem normal
        for s in range(0, r - 1):
            if ord_dummy_red[s][0] > ord_dummy_red[s+1][0]:
                ord_dummy_red[s+1], ord_dummy_red[s] = ord_dummy_red[s], ord_dummy_red[s+1]
            elif ord_dummy_red[s][0] == ord_dummy_red[s+1][0] and ord_dummy_red[s][1] > ord_dummy_red[s+1][1]:
                ord_dummy_red[s+1], ord_dummy_red[s] = ord_dummy_red[s], ord_dummy_red[s+1]
    for i in range(0, n):
        lista_ord_temp1 = list(ord_dummy_red[i])
        for j in range(2, len(lista_ord_temp1)):
            if Pi_covar.count(lista_ord_temp1[j]) != 0:
                new_indicador = True
                for k in range(j+1, len(lista_ord_temp1)):
                    if lista_ord_temp1[j] == - lista_ord_temp1[k]:
                        new_indicador = False
                        lista_ord_temp1[j], lista_ord_temp1[k] = lista_ord_temp1[k], lista_ord_temp1[j]
                if new_indicador:
                    for p in range(i+1, n):
                        lista_ord_temp2 = list(ord_dummy_red[p])
                        for q in range(2, len(lista_ord_temp2)):
                            if lista_ord_temp1[j] == - lista_ord_temp2[q]:
                                new_indicador = False
                                lista_ord_temp1[j], lista_ord_temp2[q] = lista_ord_temp2[q], lista_ord_temp1[j]
                        if not new_indicador:
                            tupla_ord_temp1 = tuple(lista_ord_temp1)
                            tupla_ord_temp2 = tuple(lista_ord_temp2)
                            ord_dummy_red.pop(i)
                            ord_dummy_red.insert(i, tupla_ord_temp1)
                            ord_dummy_red.pop(p)
                            ord_dummy_red.insert(p, tupla_ord_temp2)
                else:
                    tupla_ord_temp1 = tuple(lista_ord_temp1)
                    ord_dummy_red.pop(i)
                    ord_dummy_red.insert(i, tupla_ord_temp1)
    return ord_dummy_red
#####################################################################################################################################
def simplificacoes_da_curvatura(indicador, ordem_dummy, ordem_dummy_reduzida, pos_ord, num_elementos):
    """
    Implementa diversas simplificações associadas ao tensor de Riemann.
    """
    l = pos_ord
    n_elem = num_elementos
    pos_R = n_elem - 1
    ord_dummy = ordem_dummy
    ord_dummy_red = ordem_dummy_reduzida
    list1_temp = list(ord_dummy[l])
    count_greek = 0 #contador do número de índices gregos num dado tensor
    lista_idx_mudo, lista_idx_free = [], []
    t1_rank, n1_dif, ind1 = ord_dummy[l][0], ord_dummy[l][1], ord_dummy[l][2]
    for p in range(n1_dif + 2, len(list1_temp)): #varre todos os índices presentes na l_ésima tupla, sem contar os grad's
        if Idx_contra.count(list1_temp[p]) != 0:
            lista_idx_free.append((Idx_contra.index(list1_temp[p]), p))
            count_greek = count_greek + 1
        elif Pi_contra.count(list1_temp[p]) != 0:
            lista_idx_mudo.append((Pi_contra.index(list1_temp[p]), p))
        else:
            lista_idx_mudo.append((Pi_covar.index(list1_temp[p]), p))
    if count_greek == 2 and (list1_temp[n1_dif + 2], list1_temp[n1_dif + 3]) == (alpha, beta):
        #indica que R(alpha, beta, -pi_i, -pi_j) = 0 para quaisquer i e j, pois considera-se simetria entra alpha e beta
        indicador = 0*indicador
    elif count_greek == 2 and lista_idx_mudo[0][0] > lista_idx_mudo[1][0]:
        #exige que pi_0 venha antes de pi_1, pois R(alpha, -pi_0, beta, -pi_1) = R(alpha, -pi_1, beta, -pi_0),
        #se levarmos em conta a simetria entre alpha e beta e a identidade cíclica do tensor de Riemann
        elem_temp1 = list1_temp[lista_idx_mudo[0][1]]
        list1_temp[lista_idx_mudo[0][1]] = list1_temp[lista_idx_mudo[1][1]]
        list1_temp[lista_idx_mudo[1][1]] = elem_temp1
        tupla1_temp = tuple(list1_temp)
        indicador = 2*indicador
        delta_n = len(ord_dummy) - len(ord_dummy_red) #ajusta a posição do elemento após algumas simplifcações
        ord_dummy_red.pop(l - delta_n)
        ord_dummy_red.insert(l - delta_n, tupla1_temp)
    elif t1_rank == pos_R and count_greek == 1:
        indica_troca = False
        ind_riemann0 = list1_temp[n1_dif+2]
        ind_riemann1 = list1_temp[n1_dif+3]
        ind_riemann2 = list1_temp[n1_dif+4]
        ind_riemann3 = list1_temp[n1_dif+5]
        if n1_dif != 0 and (ind_riemann1 == - ind_riemann2 or ind_riemann1 == - ind_riemann3):
            for q in range(2, n1_dif + 2): #varre todos os índices dos grad's
                if list1_temp[q] == -list1_temp[n1_dif + 4]:
                    elem_temp2 = list1_temp[q]
                    list1_temp[q] = list1_temp[lista_idx_free[0][1]]
                    list1_temp[lista_idx_free[0][1]] = elem_temp2
                    tupla1_temp = tuple(list1_temp)
                    indicador = 2*indicador
                    delta_n = len(ord_dummy) - len(ord_dummy_red) #ajusta a posição do elemento após algumas simplifcações
                    ord_dummy_red.pop(l - delta_n)
                    ord_dummy_red.insert(l - delta_n, tupla1_temp)
                    indica_troca = True
            if not indica_troca and list1_temp[n1_dif + 2] == alpha:
                elem_temp3 = list1_temp[n1_dif + 1]
                list1_temp[n1_dif + 1] = alpha
                list1_temp[n1_dif + 2] = elem_temp3
                tupla1_temp = tuple(list1_temp)
                indicador = 2*indicador
                delta_n = len(ord_dummy) - len(ord_dummy_red) #ajusta a posição do elemento após algumas simplifcações
                ord_dummy_red.pop(l - delta_n)
                ord_dummy_red.insert(l - delta_n, tupla1_temp)
        elif ind_riemann1 != - ind_riemann2 and ind_riemann1 != - ind_riemann3 and ind_riemann2 != - ind_riemann3:
            for r in range(2, 0, -1): #Este é o algoritmo da bolha, que irá colocar as tuplas em ordem normal
                for s in range(r):
                    if lista_idx_mudo[s][0] > lista_idx_mudo[s+1][0]:
                        elem_temp4 = list1_temp[lista_idx_mudo[s][1]]
                        list1_temp[lista_idx_mudo[s][1]] = list1_temp[lista_idx_mudo[s+1][1]]
                        list1_temp[lista_idx_mudo[s+1][1]] = elem_temp4
                        temp = lista_idx_mudo[s]
                        lista_idx_mudo[s] = lista_idx_mudo[s+1]
                        lista_idx_mudo[s+1] = temp
            tupla1_temp = tuple(list1_temp)
            indicador = 2*indicador
            delta_n = len(ord_dummy) - len(ord_dummy_red) #ajusta a posição do elemento após algumas simplifcações
            ord_dummy_red.pop(l - delta_n)
            ord_dummy_red.insert(l - delta_n, tupla1_temp)
    elif t1_rank == pos_R and count_greek == 0 and list1_temp[n1_dif + 3] == -list1_temp[n1_dif + 5]:
        #impõe a equivalência obtida da identidade de Bianchi contraída no caso escalar:
        #nabla(pi_0)*R(pi_1, pi_2, pi_0, -pi_2) = (1/2)*nabla(pi_1)*R(pi_0, pi_2, -pi_0, -pi_2)
        for q in range(2, n1_dif + 2): #varre todos os índices dos grad's
            if list1_temp[q] == -list1_temp[n1_dif + 4]:
                elem_temp5 = list1_temp[q]
                list1_temp[q] = list1_temp[lista_idx_mudo[0][1]]
                list1_temp[lista_idx_mudo[0][1]] = elem_temp5
                tupla1_temp = tuple(list1_temp)
                indicador = 2*indicador
                delta_n = len(ord_dummy) - len(ord_dummy_red) #ajusta a posição do elemento após algumas simplifcações
                ord_dummy_red.pop(l - delta_n)
                ord_dummy_red.insert(l - delta_n, tupla1_temp)
    elif t1_rank == 3:
        #impõe que o divergente de F seja nulo pela identidade de Bianchi contraída: D(pi_0)*F(a,-pi_0)=0, com 'a' livre ou mudo
        for q in range(2, n1_dif + 2): #varre todos os índices dos grad's
            if list1_temp[q] == - list1_temp[n1_dif + 2] or list1_temp[q] == - list1_temp[n1_dif + 3]:
                indicador = 0*indicador
        if list1_temp[n1_dif + 2] == - list1_temp[n1_dif + 3]:
            indicador = 0*indicador ##impõe a condição de traço nulo sobre F
    return indicador, ord_dummy_red
#####################################################################################################################################
def simplificacoes_da_velocidade(indicador, ordem_dummy, ordem_dummy_reduzida, pos_ord, num_elementos):
    """
    Implementa diversas simplificações associadas ao vetor velocidade.
    """
    l = pos_ord
    n_elem = num_elementos
    pos_R = n_elem - 1
    ord_dummy = ordem_dummy
    #print(ord_dummy)
    ord_dummy_red = ordem_dummy_reduzida
    t1_rank, n1_dif, ind1 = ord_dummy[l][0], ord_dummy[l][1], ord_dummy[l][2]
    if Idx_contra.count(ind1) != 0:
        indicador = 0*indicador #impõe exigência de que a estrutura vetorial ou tensorial seja transversal
    elif Idx_contra.count(ind1) == 0:
        for p in range(l+1, len(ord_dummy)): #varre todos os 'pedaços' da estrutura tensorial além da l-ésima parte
            t2_rank, n2_dif = ord_dummy[p][0], ord_dummy[p][1]
            #considera que u(pi_0)*u(-pi_0) é uma constante que pode ser removida da estrutura tensorial
            if t2_rank == 2 and pos_R % 3 == 0 and n2_dif == 0 and ind1 == -ord_dummy[p][2]:
                indicador = 2*indicador
                ord_dummy_red.remove((2, 0, ind1))
                ord_dummy_red.remove((2, 0, -ind1))
            elif pos_R % 3 != 0 and t2_rank == 0 and n2_dif == 0 and ind1 == -ord_dummy[p][2]:
                indicador = 2*indicador
                ord_dummy_red.remove((0, 0, ind1))
                ord_dummy_red.remove((0, 0, -ind1))
            if pos_R % 3 == 0 and t2_rank != pos_R and n2_dif != 0:
                for q in range(n2_dif):
                    if ind1 == - ord_dummy[p][2 + q]:
                        indicador = 0*indicador #indica que a contração de u com as derivadas transversais nabla é zero
                    if t2_rank == 2 and ind1 == - ord_dummy[p][2 + n2_dif]:
                        indicador = 0*indicador #indica que u(pi_0) nabla...nabla*u(-pi_0) = 0, pois u(pi_0)u(-pi_0) é constante
            elif pos_R % 3 != 0 and t2_rank != 3 and t2_rank != 4 and n2_dif != 0:
                for r in range(n2_dif):
                    if ind1 == - ord_dummy[p][2 + r]:
                        indicador = 0*indicador #indica que a contração de u com as derivadas D podem ser ignoradas
                    if (t2_rank == 0 or t2_rank == 1) and ind1 == - ord_dummy[p][2 + n2_dif]:
                        indicador = 0*indicador #indica que u(pi_0) D...D*u(-pi_0) = 0, pois u(pi_0)u(-pi_0) é constante
            if (n_elem == 7 and (t2_rank == 4 or t2_rank == 5)) or (n_elem == 5 and (t2_rank == 1 or t2_rank == 2)):
                #considera a transversalidade de sigma e Omega, bem como das derivadas transversais dessas quantidades
                for q in range(2, len(ord_dummy[p])):
                    if ind1 == - ord_dummy[p][q]:
                        indicador = 0*indicador
            elif pos_R % 3 != 0 and (t2_rank == 1 or t2_rank == 2) and n2_dif != 0:
                #considera a transversalidade de sigma e Omega, bem como das derivadas transversais dessas quantidades
                for q in range(2, len(ord_dummy[p])):
                    if ind1 == - ord_dummy[p][q]:
                        indicador = 0*indicador
    return indicador, ord_dummy_red
#####################################################################################################################################
def simplifica_tensor(tensor, tipo_de_expansao):
    """
    Implementa uma série de simplificações nas estruturas tensoriais contraídas.
    Retorna um novo tensor e uma nova ordem normal com índices (reduzida).
    """
    tipo_exp = tipo_de_expansao
    if tipo_exp == 'dl':
        n_elem = 7
    elif tipo_exp == 'gz':
        n_elem = 4
    else:
        n_elem = 5
    pos_R = n_elem - 1
    ord_dummy = ordem_com_indices(tensor, tipo_exp)
    #print(ord_dummy)
    ord_dummy_red = ord_dummy.copy() #lista formada por tuplas de ordem normal com índices, reduzida após simplificações
    indicador = 1
    for l in range(len(ord_dummy)): #varre todos as tuplas que compoõem a lista de ordem normal com índices
        t1_rank, n1_dif = ord_dummy[l][0], ord_dummy[l][1]
        if len(ord_dummy[l]) > 2: #essa condição indica que as simplificações não se aplicam ao escalar Theta (sem derivadas)
            if (pos_R % 3 == 0 and t1_rank == n_elem - 1) or (pos_R % 3 != 0 and (t1_rank == 4 or t1_rank == 5)):
                indicador, ord_dummy_red = simplificacoes_da_curvatura(indicador, ord_dummy, ord_dummy_red, l, n_elem)
            elif ((pos_R % 3 == 0 and t1_rank == 2) or (pos_R % 3 != 0 and t1_rank == 0)) and n1_dif == 0:
                indicador, ord_dummy_red = simplificacoes_da_velocidade(indicador, ord_dummy, ord_dummy_red, l, n_elem)
                #print(ord_dummy_red)
            elif pos_R % 3 == 0 and (t1_rank == 0 or t1_rank == 1) and n1_dif != 0:
                for q in range(0, n1_dif):
                    ind_temp = ord_dummy[l][2+q]
                    if Idx_contra.count(ind_temp) == 0:
                        for r in range(l, len(ord_dummy)):
                            t2_rank, n2_dif = ord_dummy[r][0], ord_dummy[r][1]
                            if (t2_rank, n2_dif) == (2, 0) and ind_temp == - ord_dummy[r][2]:
                                indicador = 0*indicador
            elif (t1_rank + pos_R % 3 == 4 or t1_rank + pos_R % 3 == 5) and ord_dummy[l][n1_dif+2] == -ord_dummy[l][n1_dif+3]:
                indicador = 0*indicador #impõe a condição de traço nulo de sigma e de Omega
            #impõe condições sobre as derivadas de sigma
            elif n_elem == 7 and t1_rank == 4 and n1_dif != 0:
                #print('ord_dummy_original: ', ord_dummy)
                list1_temp = list(ord_dummy[l])
                ind_dif2 = list1_temp[n1_dif+2]
                ind_dif3 = list1_temp[n1_dif+3]
                n_contracao = 0
                pos_idx = []
                for t in range(n1_dif):
                    if list1_temp[t+2] == - ind_dif2:
                        n_contracao = n_contracao + 1
                        pos_idx.append(t+2)
                        pos_ind_fixo = n1_dif + 3
                    elif list1_temp[t+2] == -ind_dif3:
                        n_contracao = n_contracao + 1
                        pos_idx.append(t+2)
                        pos_ind_fixo = n1_dif + 2
                #print('numero de contracoes = ', n_contracao)
                if n_contracao == 2:
                    list1_temp[0] = 3
                    list1_temp[pos_idx[1]] = - list1_temp[pos_idx[0]]
                    list1_temp.pop(len(list1_temp) - 1)
                    list1_temp.pop(len(list1_temp) - 1)
                    tupla1_temp = tuple(list1_temp)
                    indicador = 2*indicador
                    delta_n = len(ord_dummy) - len(ord_dummy_red) #ajusta a posição do elemento após algumas simplifcações
                    ord_dummy_red.pop(l - delta_n)
                    ord_dummy_red.insert(l - delta_n, tupla1_temp)
                elif n_contracao == 1 and n1_dif > 1:
                    if pos_idx[0] == n1_dif + 1:
                        list1_temp[pos_idx[0]] = -list1_temp[pos_idx[0]]
                        if pos_ind_fixo == n1_dif + 3:
                            list1_temp[pos_idx[0]-1], list1_temp[pos_ind_fixo-1] = -list1_temp[pos_ind_fixo-1], list1_temp[pos_idx[0]-1]
                        else:
                            list1_temp[pos_idx[0]-1], list1_temp[pos_ind_fixo+1] = -list1_temp[pos_ind_fixo+1], list1_temp[pos_idx[0]-1]
                    else:
                        if pos_ind_fixo == n1_dif + 3:
                            list1_temp[pos_idx[0]+1], list1_temp[pos_ind_fixo-1] = list1_temp[pos_ind_fixo-1], list1_temp[pos_idx[0
                                                                                                                                  ]+1]
                        else:
                            list1_temp[pos_idx[0]+1], list1_temp[pos_ind_fixo+1] = list1_temp[pos_ind_fixo+1], list1_temp[pos_idx[0]+1]
                    n_mudos = Pi_covar.count(list1_temp[len(list1_temp)-2]) + Pi_covar.count(list1_temp[len(list1_temp)-1])
                    if n_mudos == 2 and Pi_covar.index(list1_temp[len(list1_temp)-2]) > Pi_covar.index(list1_temp[len(list1_temp)-1]):
                        elem_temporario = list1_temp[len(list1_temp)-2]
                        list1_temp[len(list1_temp)-2] = list1_temp[len(list1_temp)-1]
                        list1_temp[len(list1_temp)-1] = elem_temporario
                    tupla1_temp = tuple(reordena_lista(list1_temp, 0))
                    indicador = 2*indicador
                    delta_n = len(ord_dummy) - len(ord_dummy_red) #ajusta a posição do elemento após algumas simplifcações
                    ord_dummy_red.pop(l - delta_n)
                    ord_dummy_red.insert(l - delta_n, tupla1_temp)
            #indica que todos os termos contendo derivadas de Omega não são independentes
            elif ((n_elem == 7 and t1_rank == 5) or (n_elem == 5 and t1_rank == 2)) and n1_dif != 0:
                indicador = 0*indicador
    if indicador == 0:
        T_temporario = 0
    elif indicador == 1:
        T_temporario = reconstroi_tensor(ord_dummy, tipo_exp)
    else:
        nova_ord_dummy = reordena_indices(ord_dummy_red)
        T_temporario = reconstroi_tensor(nova_ord_dummy, tipo_exp)
    return T_temporario, ord_dummy_red
#####################################################################################################################################
def estruturas_contraidas(grau_tens, Estruturas_tensoriais, tipo_de_expansao):
    """
    Realiza todas as contrações possíveis envolvendo as estruturas tensoriais que produzem um tensor
    de determinada ordem (0, 1 ou 2). Retorna uma lista de estruturas contraídas independentes, sejam
    elas do tipo escalar, vetorial ou tensorial.
    """
    tipo_exp = tipo_de_expansao
    if tipo_exp == 'dl':
        n_elem = 7
    elif tipo_exp == 'gz':
        n_elem = 4
    else:
        n_elem = 5
    pos_R = n_elem - 1
    N = grau_tens
    Est_contraida = [] #lista que armazena as estruturas contraídas independentes de grau N
    from itertools import permutations
    for r in range(0, len(Estruturas_tensoriais)): #varre todas as estruturas tensoriais possíveis para uma dada hidrodinâmica
        #print('r = ', r)
        estrutura_nova = ordem_normal_free(Estruturas_tensoriais[r], tipo_exp)[0]
        ord_free = ordem_normal_free(Estruturas_tensoriais[r], tipo_exp)[1]
        ord_dummy = ordem_com_indices(Estruturas_tensoriais[r], tipo_exp)
        (count_u, count_F, n_dif_F, count_R, n_dif_R) = (0, 0, 0, 0, 0) #contadores dos u's, de R's e F's e de suas derivadas
        for s in range(len(ord_dummy)):
            if ((pos_R % 3 == 0 and ord_dummy[s][0] == 2) or (pos_R % 3 != 0 and ord_dummy[s][0] == 0)) and ord_dummy[s][1] == 0:
                count_u = count_u + 1
            elif ord_dummy[s][0] == pos_R:
                count_R = count_R + 1
                n_dif_R = n_dif_R + ord_dummy[s][1]
            elif tipo_exp == 'cf' and ord_dummy[s][0] == 3:
                count_F = count_F + 1
                n_dif_F = n_dif_F + ord_dummy[s][1]
        lista_original = Estruturas_tensoriais[r].get_indices() #lista formada pelos índices da r1_ésima estrutura tensorial
        #testa se é possível formar um elemento não-nulo para um dado número de u's
        if pos_R % 3 == 0 and count_u > 2*count_R + n_dif_R:
            tensor_temp = 0
        elif pos_R % 3 != 0 and count_u > count_R + count_F + n_dif_R + n_dif_F:
            tensor_temp = 0
        else:
            lista = []
            for i in range(len(lista_original)):
                lista.append(i)
            perm = permutations(lista)
            lista_completa = [] #armazena as listas de índices que produzirão tensores distintos
            for p in perm:
                p_lista = []
                for j in range(len(p)):
                    p_lista.append(p[j])
                if len(p_lista) == 0: #condição em que a estrutura é um escalar, como o Theta sem derivadas
                    lista_completa = p_lista
                    coeficiente = 1
                    if len(Estruturas_tensoriais[r].get_indices()) == 0 and N == 2: #Escalar solitário não produz um tensor TST
                        coeficiente = 0
                    tensor_temp = coeficiente*estrutura_nova
                    #tensor_temp = coeficiente*Estruturas_tensoriais[r]
                    if tensor_temp != 0: #confere se o escalar já consta da lista de estruturas contraídas
                        data = tensor_temp.args[0]
                        if data.is_Symbol or data.is_Add:
                            tensor_temp = tensor_temp/data
                        if not (compara_tensor_lista(tensor_temp, Est_contraida)):
                            Est_contraida.append(tensor_temp)
                else:
                    #canonicaliza a lista permutada
                    p_lista_canonica = tensor_forma_canonica(N, p_lista, ord_free, False, tipo_exp)
                    if isinstance(p_lista_canonica, list):
                        p_tupla = tuple(p_lista_canonica)
                        if lista_completa.count(p_tupla) == 0:
                            lista_completa.append(p_tupla)
            for k in range(len(lista_completa)): #varre todas as listas de índices que podem formar estruturas novas para dado r1
                lista_reordenada = reordena_lista(list(lista_completa[k]), 0) #impõe que alpha esteja à esquerda de beta
                Ind_tuplas = []
                for l in range(len(lista_original)):
                    Ind_tuplas.append((lista_original[l], lista_reordenada[l]))
                tensor_new = estrutura_nova.substitute_indices(*Ind_tuplas) #tensor formado c/ base em simetrias usuais
                #print('tensor_new = ', tensor_new)
                tensor_temp, ord_dummy_new = simplifica_tensor(tensor_new, tipo_exp) #tensor que resulta impondo simplificações extras
                #print('tensor_temp = ', tensor_temp)
                if N == 2 and tensor_temp != 0: #uma segunda camada de simplificações que leva em conta o caráter TST de N = 2
                    ord_normal = ordem_com_indices(tensor_temp, tipo_exp)
                    lista_indices = tensor_temp.get_indices()
                    list_new = []
                    for i in range(len(lista_indices)): #transforma-se aqui a lista de índices numa sequência numérica
                        if Pi_contra.count(lista_indices[i]) != 0:
                            p_contra = Pi_contra.index(lista_indices[i])
                            list_new.append(N + 2*p_contra)
                        elif Pi_covar.count(lista_indices[i]) != 0:
                            p_covar = Pi_covar.index(lista_indices[i])
                            list_new.append(N + 2*p_covar + 1)
                        elif lista_indices[i] == alpha:
                            list_new.append(0)
                        else:
                            list_new.append(1)
                    lista_temp = tensor_forma_canonica(N, list_new, ord_normal, True, tipo_exp)
                    if isinstance(lista_temp, list):
                        conta_indice = 0
                        for j in range(len(ord_dummy_new)):
                            lista_ord_dummy = list(ord_dummy_new[j])
                            num_indice = 0
                            for m in range(2, len(lista_ord_dummy)):
                                lista_ord_dummy[m] = lista_temp[conta_indice + m - 2]
                                num_indice = num_indice + 1
                            conta_indice = conta_indice + num_indice
                            tupla_ord_dummy = tuple(lista_ord_dummy)
                            ord_dummy_new.pop(j)
                            ord_dummy_new.insert(j, tupla_ord_dummy)
                        tensor_temp = reconstroi_tensor(ord_dummy_new, tipo_exp)
                    else:
                        tensor_temp = 0
                if tensor_temp != 0:
                    data = tensor_temp.args[0]
                    if data.is_Symbol or data.is_Add:
                        tensor_temp = tensor_temp/data
                    if not (compara_tensor_lista(tensor_temp, Est_contraida)):
                        Est_contraida.append(tensor_temp)
                        display(tensor_temp)
    return Est_contraida
#####################################################################################################################################
tipo_de_expansao = 'gz'
ordem_da_expansao = 3
curvatura = 1
lista_ingredientes, lista_M = ingredientes(ordem_da_expansao, curvatura, tipo_de_expansao)
for k in range(len(lista_ingredientes[ordem_da_expansao])):
    print(k)
    display(inverte_indices(lista_ingredientes[ordem_da_expansao][k]))
#####################################################################################################################################
grau_tensor = 2
Estruturas_gerais = estruturas_tensoriais(ordem_da_expansao, grau_tensor, lista_ingredientes, lista_M, tipo_de_expansao)
for i in range(0, len(Estruturas_gerais)):
    print(i)
    display(inverte_indices(Estruturas_gerais[i]))
#####################################################################################################################################
est_tens_contraidas = estruturas_contraidas(grau_tensor, Estruturas_gerais, tipo_de_expansao)
for j in range(len(est_tens_contraidas)):
    print(j)
    display(est_tens_contraidas[j])
