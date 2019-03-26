%% Initialization
close all
clear all
clc 

%% Loading and Visualizing Data

% Load Training Data
load('train_data.mat');

%% Training and Validation Datasets 
rand('seed',13);
valid_idx = randperm(size(all_images_normalized, 1),round(0.2*size(all_images_normalized, 1)));
train_idx = setdiff(1:size(all_images_normalized, 1),valid_idx);
train_images = all_images_normalized(train_idx,:);
validation_images = all_images_normalized(valid_idx,:);
train_labels = label(train_idx,:);
validation_labels = label(valid_idx,:);
%% Validation Results
PCA_features = 12;
[A ,Y ,eigen_values] = PCA_transformation(train_images, PCA_features);
Yva = validation_images * A;

%% KNN
mdl = fitcknn(Y,train_labels,'NumNeighbors',1,...
                                'Distance','euclidean',...
                                'DistanceWeight','equal','Exponent','', 'StandardizeData', 1);
pred = predict(mdl,Yva);
plotconfusion(ind2vec(pred'),ind2vec(validation_labels')); 
