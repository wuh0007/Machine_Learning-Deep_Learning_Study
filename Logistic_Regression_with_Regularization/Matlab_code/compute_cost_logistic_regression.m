function [J] = compute_cost_logistic_regression(theta,x_norm,y)
 % Compute cost for logistic regression 
 
    % Initialize some useful values
    m = size(x_norm,1); % number of training examples
    
    % Compute cost
    h = compute_sigmoid(x_norm*theta);
    J = -1/m*sum(y.*log(h) + (1 - y).*log(1 - h));
    
end

