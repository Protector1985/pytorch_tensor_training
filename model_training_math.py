import torch

temperature_readings_celsius = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
termperature_readings_unknown_units = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

tensor_celsius_scale = torch.tensor(temperature_readings_celsius)
tensor_unknown_scale = torch.tensor(termperature_readings_unknown_units)

#assumption calculation
#weight = how much do inputs influence outputs
#bias = if input is 0, what would the output be?
#temperature_readings_celsius = weight * temperature_readings_unknown_units + bias


def model(temp_unknown, weight, bias):
    return weight * temp_unknown + bias

def loss_function(temp_predicted, temp_celsius):
    loss_calc = (temp_predicted - temp_celsius)**2
    return loss_calc.mean()

def derivative_loss_function(temp_predicted, temp_celsius):
    d_squared_diffs = 2 * (temp_predicted - temp_celsius) / temp_predicted.numel()
    return d_squared_diffs

def dModel_dw(temp_unknown, w, b):
    return temp_unknown

def dModel_db(temp_unknown, w, b):
    return 1.0

#performs rthe gradient descent step
def gradient_descent_function(temp_unknown, temp_celsius, temp_predicted, weight, bias):
    dlossDTP = derivative_loss_function(temp_predicted, temp_celsius)
    dlossDWeight = dlossDTP * dModel_dw(temp_unknown, weight, bias)
    dlossDBias = dlossDTP * dModel_db(temp_unknown, weight, bias)
    return torch.stack([dlossDWeight.sum(), dlossDBias.sum()])
    

w = torch.ones(())
b = torch.zeros(())

temp_prediction = model(tensor_unknown_scale, w, b)
loss = loss_function(temp_prediction, tensor_celsius_scale)


def training_loop(num_epochs, learning_rate, params, temp_unknown, temp_celsius):
    for epoch in range(1, num_epochs + 1):
        weight, bias = params
        
        temp_predicted = model(temp_unknown, weight, bias)
        #forward propogation
        loss = loss_function(temp_predicted, temp_celsius)
        gradient_descent = gradient_descent_function(temp_unknown, temp_celsius, temp_predicted, weight, bias)
        #back propogation
        params = params - learning_rate * gradient_descent
        
        print('Epoch : %d, Loss: %f,' % (epoch, float(loss)))
        
        return params

delta = 0.1
learning_rate = 1e-2
temp_pred_weight1= model(tensor_unknown_scale, w + delta, b)
temp_pred_weight2= model(tensor_unknown_scale, w - delta, b)
temp_pred_bias1 = model(tensor_unknown_scale, w, b + delta)
temp_pred_bias2 = model(tensor_unknown_scale, w, b - delta)

weight_loss_rate_of_change = (loss_function(temp_pred_weight1, tensor_celsius_scale) - loss_function(temp_pred_weight2, tensor_celsius_scale)) / (2.0 * delta)
bias_loss_rate_of_change = (loss_function(temp_pred_bias1, tensor_celsius_scale) - loss_function(temp_pred_bias2, tensor_celsius_scale)) / (2.0 * delta)
weight = w - learning_rate * weight_loss_rate_of_change
bias = b - learning_rate * bias_loss_rate_of_change

training_loop(num_epochs=100, learning_rate=1e-2, params=torch.tensor([1.0, 0.0]), temp_unknown=tensor_unknown_scale, temp_celsius=tensor_celsius_scale)