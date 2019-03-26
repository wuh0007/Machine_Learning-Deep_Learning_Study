%% Initialization
close all
clear all
clc 

%% Loading data

% Load Training Data
load('train_data.mat');
m = size(all_images_normalized, 1);

%% Training and Validation Datasets 
rand('seed',13);
valid_idx = randperm(size(all_images_normalized, 1),round(0.2*size(all_images_normalized, 1)));
train_idx = setdiff(1:size(all_images_normalized, 1),valid_idx);
train_images = all_images_normalized(train_idx,:);
validation_images = all_images_normalized(valid_idx,:);
train_labels = label(train_idx,:);
validation_labels = label(valid_idx,:);
%% Validation Results
PCA_features = 52;
[A ,Y ,eigen_values] = PCA_transformation(train_images, PCA_features);
Yva = validation_images * A;

%% Setup the parameters 
input_layer_size  = size(Y, 2);  % 256x16 input Images of Digits
hidden_layer_size = round(0.66 * size(Y, 2)) + 9;   % 2/3 * input layer units + num_labels for hidden units
num_labels = 9;          % 9 labels, from 1 to 9
lambda = 0;                % regularization factor

%% Initializing Pameters

% Radomly initialize the neural network parameters
random_initial = 0.12;
initial_theta1 = random_initial*2*rand(hidden_layer_size,input_layer_size+1)-random_initial;
initial_theta2 = random_initial*2*rand(num_labels,hidden_layer_size+1)-random_initial;
initial_theta = [initial_theta1(:) ; initial_theta2(:)];
%% Training Neural Network

% Return the cost and grad values
options = optimset('MaxIter', 200);
costFunction = @(p) nn_cost_function(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Y, train_labels, lambda);

% Learning Process
[params, cost] = fmincg(costFunction, initial_theta, options);

% Obtain Theta1 and Theta2 back from nn_params
theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%% Visualize Weights 
% displayData(theta1(:, 2:end));
%% Prediction
p = determine_output(theta1, theta2, Yva);
fprintf('Training Set Accuracy: %f\n', mean(double(p == validation_labels)) * 100);
plotconfusion(ind2vec(p'),ind2vec(validation_labels')); 