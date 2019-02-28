%% Initialization
close all
clear all
clc 

%% Loading and Visualizing Data

% Load Training Data
load('Sample_MNIST.mat');
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel,:));
%% Training and Validation Datasets 
load('indices.mat');
train_images = X(train_idx,:);
validation_images = X(valid_idx,:);
train_labels = y(train_idx,:);
validation_labels = y(valid_idx,:);

% rand('seed',13);
% sel2 = randperm(size(X, 1));
% train = sel2(1:3500);
% valid = sel2(3501:end);
% train_images = X(train,:);
% validation_images = X(valid,:);
% train_labels = y(train,:);
% validation_labels = y(valid,:);
%% Validation Results
PCA_features = 20;
[A ,Y ,eigen_values] = PCA_transformation(train_images, PCA_features);
Yva = validation_images * A;

%% KNN
mdl = fitcknn(train_images,train_labels,'NumNeighbors',10,...
                                'Distance','euclidean',...
                                'DistanceWeight','squaredinverse');
pred = predict(mdl,validation_images);
plotconfusion(ind2vec(pred'),ind2vec(validation_labels')); 
