function [A ,Y ,eigen_values] = PCA_transformation(train_images, N)
% PCA (Principal Component Analysis) on dataset
% compute Transformation Matrix, Train image after PCA Transformation, and
% Eigen Values based on Train images and number of PCA features.
cov_M = cov(train_images);
[eigen_vector,eigen_values] = eig(cov_M);
[eigen_values, indices] = sort(diag(eigen_values),'descend');
eigen_vector = eigen_vector(:,indices);

A = eigen_vector(:,1:N);
Y = train_images * A;
end
