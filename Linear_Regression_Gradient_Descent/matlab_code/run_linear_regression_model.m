clear all
close all
clc

% load data
data = load('Sample Data.txt');
% Grab the relevant data, scale the predictor variable, 
% and add a column of 1s for the gradient descent
y = data(:,end);
x_norm = normalize_features(data(:,1:end-1));
x_ones = ones(size(x_norm,1),1);
x_norm = [x_ones x_norm];

% Other parameters, initialize n zeros for theta
theta = zeros(size(x_norm,2),1);
alpha = 0.01;
iterations = 1000;
% Call the Gradient descent function
[J,theta] = gradient_descent_lr_multi_variable(theta,x_norm,y,alpha,iterations)

%Plot the result
figure;
plot(1:iterations, J);
xlabel("Iteration #");
ylabel("Cost Values");
title("Cost Function");


data = load('airfoil_self_noise.txt');
y = data(:,end);
x_norm = normalize_features(data(:,1:end-1));
x_ones = ones(size(x_norm,1),1);
x_norm = [x_ones x_norm];

theta = zeros(size(x_norm,2),1);
alpha = 0.01;
iterations = 1000;

[J,theta] = gradient_descent_lr_multi_variable(theta,x_norm,y,alpha,iterations)

figure;
plot(1:iterations, J);
xlabel("Iteration #");
ylabel("Cost Values");
title("Cost Function");