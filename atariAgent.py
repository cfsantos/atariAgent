
# coding: utf-8
# OpenAI Gym (https://gym.openai.com) possui uma grande variedade de exemplos e jogos para treinar um agente.
# Criaremos aqui uma rede neural que, dado o estado do jogo (dois estados consecutivos), gera os Q-Values para cada movimento seguinte. 
# O movimento com maior Q-Value é escolhido e executado no jogo.
# Veja o formalismo em https://www.nervanasys.com/demystifying-deep-reinforcement-learning/


# Inicio: libraries, parametros, rede neural e etc...

from keras.models import Sequential      
from keras.layers import Dense, Flatten  
from collections import deque               # Para armazenar os movimentos

import numpy as np
import gym                                  # Para treinar a nossa rede neural
env = gym.make('Assault-v0')                # Escolha o jogo (necessário prestar atencao as entradas, algumas sao um pouco diferentes)
         

import random     
print  env.observation_space.shape, env.action_space.n

# Criacao da rede neural: Entrada - dois estados consecutivos do jogo. Saida - Q-Values para os possíveis movimentos
model = Sequential()
model.add(Dense(20,input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
model.add(Flatten())       
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(env.action_space.n, init='uniform', activation='linear'))   

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Parametros
D = deque()                              # Registrador onde as acoes do jogo serao armazenadas

observetime = 100000                     # Numero de vezes em que executaremos acoes e observacoes
epsilon = 0.7                            # Chances de executar um movimento aleatorio# Probability of doing a random move
gamma = 0.9                              # Desconto nas recompensas do futuro # Discounted future reward.
mb_size = 500                            # Tamanho do minibatch


# Primeiro Passo: Saber o que cada acao faz (Observacao)

# Inicia o jogo
observation = env.reset()                    
obs = np.expand_dims(observation, axis=0)     
state = np.stack((obs, obs), axis=1)
done = False
for t in range(observetime):
    
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, env.action_space.n, size=1)[0]
    else:
        # Predicao dos Q-values 
        Q = model.predict(state)  
        # Executa o movimento com o maior Q-Value        
        action = np.argmax(Q)             

    # Verifica o estado do jogo, recompensas e outras informacoes apos executar a acao    
    observation_new, reward, done, info = env.step(action)     
    obs_new = np.expand_dims(observation_new, axis=0)          

    # Atualiza a entrada com o novo estado do jogo
    state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     

    # Armazena o estado e as suas consequencias da acao
    D.append((state, action, reward, state_new, done))   

    # Atualiza o estado      
    state = state_new         

    if done:
        # Reinicia o jogo se finalizada as observacoes
        env.reset()           

        # (Formatacao) Faz a observacao ser o primeiro elemento do batch de entrada
        obs = np.expand_dims(observation, axis=0)     
        state = np.stack((obs, obs), axis=1)
print('Fim das observacoes')


# Segundo passo: aprendendo a partir das observacoes (Replay das experiencias)

# Coleta amostra de alguns movimentos
minibatch = random.sample(D, mb_size)                              

inputs_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros(inputs_shape)
targets = np.zeros((mb_size, env.action_space.n))

print mb_size

for i in range(0, mb_size):
    state = minibatch[i][0]
    action = minibatch[i][1]
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]
    
    #Equacao de Bellman para a Funcao Q
    inputs[i:i+1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)
    
    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)


    # Treina a rede para produzir os Q-Values
    model.train_on_batch(inputs, targets)

print('Fim do Treinamento')


# Terceiro passo: Jogar

observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
i = 0
while not done:
    #inicia a renderezacao do jogo
    env.render()                    
    Q = model.predict(state)        
    action = np.argmax(Q)         
    observation, reward, done, info = env.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
    tot_reward += reward
    i+=1

    #Dependendo do jogo, pode ocorrer de haver muitas recompensas de uma vez só. Esse if garante que haja impressao a cada 100 passos
    if (i % 100) == 0:
        print tot_reward
print('Game over! Pontuacao final: {}'.format(tot_reward))