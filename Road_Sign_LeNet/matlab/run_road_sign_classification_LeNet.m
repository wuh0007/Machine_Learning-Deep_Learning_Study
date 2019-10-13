%% Road Sign Classification
% Author: Barath Narayanan
% Date: 04/05/2019


%% Read and Visualize Different Categories

% Clear Workspace
clear all; close all;clc;

% Image Datastore
imds=imageDatastore('E:\Road Sign\Train Images\',...
    'IncludeSubFolders',1,'Labelsource','foldernames');

% Count of Each Label
total_count=countEachLabel(imds);
disp(total_count);

% Double of Labels
labels=double(imds. Labels);
figure;
bar(total_count.Count);
title('Distribution of Dataset');
pause

% % Visualize Each Category
% for idx=1:length(unique(labels))
%     
%     % Determine each category Index
%     cat_idx=find(labels==idx,1,'first');
%     
%     % Read the Image
%     I=readimage(imds,cat_idx);
%     
%     
%     % Visualize the image
%     figure(1);
%     imagesc(I);
%     title(sprintf('Category %d',idx));
%     pause;
%     
% end
% imds.ReadFcn=@(filename)preprocess_image(filename);
%% Train and Validation Datasets

% Split Dataset - Training and Validation
[imdsTrain,imdsValid]=splitEachLabel(imds,0.9);

% Count of Each Label
train_count=countEachLabel(imdsTrain);


% Count of Each Label
valid_count=countEachLabel(imdsValid);

figure;
bar([train_count.Count,valid_count.Count]);
legend('Train','Validation')

%% CNN Network
img_size=[32 32 1];

layers=[
    % Input Image Layer
    imageInputLayer([img_size])
    convolution2dLayer(3,64,'Padding','same');
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(3,128,'Padding','same');
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
       
    convolution2dLayer(3,256,'Padding','same');
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,512,'Padding','same');
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(128);
    dropoutLayer(0.5);
    fullyConnectedLayer(length(unique(labels)));
    softmaxLayer
    classificationLayer;   
  ]  


% Training Options
options=trainingOptions('adam','MaxEpochs',2,'MiniBatchSize',64,'ValidationData',imdsValid,'ValidationFrequency',100,'Plots','training-progress','ValidationPatience',6);


%% Data Augumentation
imageAugumenter=imageDataAugmenter('RandRotation',[-10 10],'RandXTranslation',[-3 3],'RandYTranslation',[-3 3]);

augimds = augmentedImageDatastore(img_size,imdsTrain,'DataAugmentation',imageAugumenter);

% Train the Network
net = trainNetwork(augimds,layers,options);

%% Test the Network

imdsTest=imageDatastore('E:\Road Sign\Train Images\',...
                'LabelSource','foldernames','IncludeSubFolders',1);
imdsTest.ReadFcn=@(filename)preprocess_image(filename);

% Predict Test Labels
[predicted_labels,posterior] = classify(net,imdsTest);



