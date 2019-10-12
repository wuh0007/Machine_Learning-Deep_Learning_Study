
%% Initialization
close all
clear all
clc 

%% Setup the parameters 
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10  , '0' mapped as '10' 
lamta = 0;                % regularization factor
%% Loading and Visualizing Data

% Load Training Data
load('Sample_MNIST.mat');
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X,1));
sel = sel(1:100);
displayData(X(sel,:));
pause
%% Initializing Pameters

% Radomly initialize the neural network parameters
random_initial = 0.12;
initial_theta1 = random_initial*2*rand(hidden_layer_size,input_layer_size+1)-random_initial;
initial_theta2 = random_initial*2*rand(num_labels,hidden_layer_size+1)-random_initial;
initial_theta = [initial_theta1(:) ; initial_theta2(:)];
%% Training Neural Network

% Return the cost and grad values
options = optimset('MaxIter', 50);
costFunction = @(p) nn_cost_function(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lamta);

% Learning Process
[params, cost] = fmincg(costFunction, initial_theta, options);

% Obtain Theta1 and Theta2 back from nn_params
theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%% Visualize Weights 
displayData(theta1(:, 2:end));
%% Prediction
p = determine_output(theta1, theta2, X);
fprintf('Training Set Accuracy: %f\n', mean(double(p == y)) * 100);