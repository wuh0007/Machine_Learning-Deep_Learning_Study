%% Initialization
close all
clear all
clc 

%% Image Datastore
imds = imageDatastore('E:\Road Sign\Train Images\',...
    'IncludeSubFolders',1,'Labelsource','foldernames');

%% Dataset count
total_count = countEachLabel(imds)

figure;
bar(total_count.Count)
title('Dataset Distribution');
xlabel('Class #');
ylabel('Count');

%% Convert Categorial to Double
labels = double(imds.Labels);

%% For Loop
num_classes = length(unique(labels));

% for idx = 1:num_classes
%     
%     % Find Command
%     cat_idx = find(labels==idx,1,'first');
% 
%     % Read the Image
%     I = readimage(imds, cat_idx);
%     
%     figure;
%     imagesc(I);
%     title(sprintf('class : %d',idx));
%     
%     size(I)
%     pause;
% end

imds.ReadFcn = @(filename)preprocess_image_roadsign(filename);

train_percentage = 0.9;
[imdsTrain, imdsValid] = splitEachLabel(imds, train_percentage);

% Count Labels
train_count=countEachLabel(imdsTrain);
valid_count=countEachLabel(imdsValid);

figure;
bar([train_count.Count,valid_count.Count])
legend('Train Dataset','Validation Dataset');
xlabel('Class #');
ylabel('Count');

%% CNN

% layers=[
%     imageInputLayer([32 32 3]);
%     convolution2dLayer(3,8);
%     batchNormalizationLayer;
%     reluLayer;
%     maxPooling2dLayer(2,'Stride',2);
%     
%     
%     convolution2dLayer(3,16);
%     batchNormalizationLayer;
%     reluLayer;   
%     maxPooling2dLayer(2,'Stride',2);
%     
%     fullyConnectedLayer(150)
%     fullyConnectedLayer(num_classes)
%     
%     softmaxLayer;
%     classificationLayer;
%     ];
 
% % layers=[
% %     imageInputLayer([32 32 3]);
% %     convolution2dLayer(3,64,'Padding','same');
% %     batchNormalizationLayer;
% %     reluLayer;
% %     maxPooling2dLayer(2,'Stride',2);
% %     
% %     
% %     convolution2dLayer(3,128,'Padding','same');
% %     batchNormalizationLayer;
% %     reluLayer;   
% %     maxPooling2dLayer(2,'Stride',2);
% %     
% %     convolution2dLayer(3,256,'Padding','same');
% %     batchNormalizationLayer;
% %     reluLayer;   
% %     maxPooling2dLayer(2,'Stride',2);    
% %     
% %     convolution2dLayer(3,512,'Padding','same');
% %     batchNormalizationLayer;
% %     reluLayer;   
% %     maxPooling2dLayer(2,'Stride',2);    
% %     
% %     
% %     fullyConnectedLayer(150)
% %     fullyConnectedLayer(num_classes)   
% %     softmaxLayer;
% %     classificationLayer;
% %     ];

 %use analyzenetwork(layers) to check architecture in Matlab2018B or later
 
% %  options = trainingOptions('adam', 'MiniBatchSize', 64, 'MaxEpochs', 2,...
% %      'ValidationData', imdsValid, 'ValidationFrequency', 100,...
% %      'plots', 'training-progress','Verbose',1,'ValidationPatience', 5)
% %  
% %  net = trainNetwork(imdsTrain,layers,options);