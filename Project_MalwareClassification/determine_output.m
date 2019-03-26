function p = predict(theta1, theta2, X)
% Predict the label of an input given a trained neural network
%   p = predict(theta1, theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (theta1, theta2)

% set up useful values
m = size(X, 1);
p = zeros(m, 1);

h1 = compute_sigmoid([ones(m, 1) X] * theta1');
h2 = compute_sigmoid([ones(m, 1) h1] * theta2');
[dummy, p] = max(h2, [], 2);

end
