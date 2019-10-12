function [J grad] = nn_cost_function(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% cost function and gradient descent for neural network

%reshape theta back into weights theta1 & theta2
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);% Setup some useful variables

%initilize following variables
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));
delta1 = zeros(size(theta1));
delta2 = zeros(size(theta2));

%vectorize the y values
y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

% feedforward
a1 = [ones(m, 1) X];
z2 = a1*theta1';
a2 = [ones(size(z2, 1), 1) compute_sigmoid(z2)];
z3 = a2*theta2';
a3 = compute_sigmoid(z3);
h = a3;

% calculte penalty
% p = sum(sum(theta1(:, 2:end).^2, 2))+sum(sum(theta2(:, 2:end).^2, 2));
% + lambda*p/(2*m

% calculate J
J = sum(sum((-y).*log(h) - (1-y).*log(1-h), 2))/m;

%back prop
% calculate sigmas
sigma3 = a3-y;
sigma2 = (sigma3*theta2).* (compute_sigmoid([ones(size(z2, 1), 1) z2]) .* (1-compute_sigmoid([ones(size(z2, 1), 1) z2])));
sigma2 = sigma2(:, 2:end);

% accumulate gradients
delta_1 = (sigma2'*a1);
delta_2 = (sigma3'*a2);

% calculate regularized gradient
theta1_grad = delta_1/m;
theta2_grad = delta_2/m;

% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:)];

end
