function [Iout] = preprocess_image(filename)
% This function is used to resize all images to a common size and
% preprocess using histogram equalization
%
% 
I=imread(filename);
I=rgb2gray(I);
I=histeq(I);

Iout=imresize(I,[32 32]);


end


