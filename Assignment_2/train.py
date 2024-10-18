import numpy as np
import json

#init weights with random values
weights = np.random.rand(784, 10) * 0.001
biases = np.zeros(10)
#learning rate suggested online
learning_rate = 0.001
batch_count = 100

#calculates with softmax function the output of the network for one batch 
def compute_output(batch):
    weighted_sums = batch.dot(weights) + biases
    #exponential sum
    exp_weighted_sums = np.exp(weighted_sums - np.max(weighted_sums, axis=1, keepdims=True))
    return exp_weighted_sums / np.sum(exp_weighted_sums, axis=1, keepdims=True)

def train(epochs, data, labels):
    global weights, biases

    for epoch in range(epochs):
        start_point = epoch * batch_count
        end_point = (epoch+1) * batch_count
        data_batch = []
        labels_batch = []
        for data_index in range(start_point, end_point):
            data_batch.append(data[data_index])
            current_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            current_label[labels[data_index]] = 1
            labels_batch.append(current_label)
        np_batch = np.array(data_batch)
        target = np.array(labels_batch)
        network_outputs = compute_output(np_batch)
        #claculate gradient loss
        gradient_loss = target - network_outputs
        #update weights an biases
        weights = weights + learning_rate * (np_batch.T.dot(gradient_loss) / 100) #divide by batch size to rescale the results
        biases = biases + learning_rate * np.mean(gradient_loss, axis=0) #calculate mean instead of dividing by 100
    #saving data to json
    wheights_list = weights.tolist()
    biases_list = biases.tolist()
    calculated_data = {
        "weights": wheights_list,
        "biases": biases_list
    }
    json_name = f'calculated_data_{epochs}.json'
    with open(json_name, 'w') as json_file:
        json.dump(calculated_data, json_file, indent=1)
