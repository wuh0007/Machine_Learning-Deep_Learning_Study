function [Iout] = preprocess_image_roadsign(filename)
% This function preprocess the road sign images by reshaping and histogram
% equalization
%
I = imread(filename);
I = imresize(I,[32 32]);
I = rgb2gray(I);
Iout = histeq(I);


end

