function [J] = compute_cost_logistic_regression_regularized(theta,x_norm,y,lamta)
 % Compute cost for logistic regression 
 
    % Initialize some useful values
    m = size(x_norm,1); % number of training examples
    
    % Compute cost
    h = compute_sigmoid(x_norm*theta);

    J = -1/m*sum(y.*log(h) + (1 - y).*log(1 - h)) + (lamta/(2*m)) * sum(theta(2:end).^2);
    
end
