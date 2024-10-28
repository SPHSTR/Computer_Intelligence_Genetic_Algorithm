import numpy as np
import matplotlib.pyplot as plt


def Load_data(file = 'wdbc.txt'):
    data_output = []
    data_input = []


    with open(file,'r') as file:
        for line in file :
            data = line.strip().split(',')

            if data[1] ==  'M':
                data_output.append(1)
            elif data[1] == "B":
                data_output.append(0)

            data_input.append(data[2:])

    return data_output , data_input

def Sigmoid(x):
    return 1 /(1 + np.exp(-x))

def Init_population(population_size,input_size,hidden_size,output_size):
    length = input_size * hidden_size + hidden_size * output_size

    return np.random.uniform(-1,1,(population_size,length))

def Init_Weight(chromosome,input_size,hidden_size,output_size):
    input_weight = chromosome[:input_size*hidden_size].reshape((input_size,hidden_size))
    hidden_weight = chromosome[input_size*hidden_size:(input_size+hidden_size)*(hidden_size+output_size)].reshape((hidden_size,output_size))

    return input_weight , hidden_weight

def Feed_forward(inputs,input_weight,hidden_weigh):
    hidden_input = np.dot(inputs,input_weight)
    hidden_output = Sigmoid(hidden_input)

    final_input = np.dot(hidden_output,hidden_weigh)
    final_output = Sigmoid(final_input)

    return final_input

def Fitness_function(inputs,indv,chromosome,input_size,hidden_size,output_size):
    input_weight , hidden_weigth = Init_Weight(chromosome,input_size,hidden_size,output_size)
    output = Feed_forward(inputs,input_weight,hidden_weigth)
    fitness = np.sum(np.argmax(output,axis=1) == np.argmax(indv,axis=1)) / len(indv)

    return fitness

def Crossing_over(P1,P2):
    Crossing_site = np.random.randint(len(P1))
    child = np.concatenate((P1[:Crossing_site],P2[Crossing_site:]))

    return child

def Mutate(chromosome,rate):
    P = np.random.rand(len(chromosome)) < rate
    chromosome[P] += np.random.uniform(-0.1,0.1,np.sum(P))

    return chromosome

def Genetic_Algorithm(inputs,indv,input_size,hidden_size,output_size,population_size,generation,rate):
    population = Init_population(population_size,input_size,hidden_size,output_size)

    for generations in range(generation):
        fitness_array = []

        for chromosome in population:
            fitness = Fitness_function(inputs,indv,chromosome,input_size,hidden_size,output_size)
            fitness_array.append(fitness)

        best_index = np.argmax(fitness_array)
        best_chromosome = population[best_index]
        best_fitness = fitness_array[best_index]

        print(f"Generation {generations + 1}, Accuracy: {best_fitness * 100:.2f}%")

        select_index = np.argsort(fitness_array)[-int(0.2*population_size):]
        select_population = population[select_index]

        new_poppulation = []

        for _ in range(population_size - len(select_population)):
            P1 = select_population[np.random.randint(len(select_population))]
            P2 = select_population[np.random.randint(len(select_population))]
            child = Mutate(Crossing_over(P1,P2),rate)
            new_poppulation.append(child)

        population = np.vstack((select_population,new_poppulation))

    return best_chromosome

def Cross_Validaion(inputs,indv,input_size,hidden_size,output_size,population_size,generation,rate,k=10):
    fold_size = len(inputs) // k
    
    fitness_array = []
    best_hidden_size = []
    best_chromosomes = []

    for fold in range(k):
        start_idex = fold * fold_size
        end_index = (fold + 1) * fold_size

        Validation_input = inputs[start_idex:end_index]
        Validation_indv = indv[start_idex: end_index]
        
        train_input = np.concatenate([inputs[:start_idex] , inputs[end_index:]])
        train_indv = np.concatenate([indv[:start_idex] , indv[end_index:]])

        best_chromosome = Genetic_Algorithm(train_input,train_indv,input_size,hidden_size[fold],output_size,population_size,generation,rate)

        fitness = Fitness_function(Validation_input,Validation_indv,best_chromosome,input_size,hidden_size[fold],output_size)
        fitness_array.append(fitness)
        best_hidden_size.append(hidden_size[fold])
        best_chromosomes.append(best_chromosome)

        print(f"Fold {fold + 1} Average Accuracy: {fitness * 100:.2f}%")

    mean_accurate = np.mean(fitness_array)
    print(f"\nAverage Accuracy: {mean_accurate * 100:.2f}%")


data_output , data_input = Load_data()

data_input = np.array(data_input,dtype=float)
data_output = np.array(data_output)

indv = np.zeros((len(data_output),2))
indv[np.arange(len(data_output)),data_output] = 1

input_size = len(data_input[0])
hidden_size = [1,2,3,4,5,6,7,8,9,10]
output_size = len(np.unique(data_output))
population_size = 50
generation = 10
Mutate_rate = 0.1

Cross_Validaion(data_input,indv,input_size,hidden_size,output_size,population_size,generation,Mutate_rate)
