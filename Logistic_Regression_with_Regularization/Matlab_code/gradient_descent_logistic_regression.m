function [J,theta] = gradient_descent_logistic_regression(theta,x_norm,y,alpha,num_iters)
% gradient descent for logistic regression 
    m = size(x_norm,1);
    
    for idx = 1:num_iters  
        % final theta vector size n x 1 
        theta = theta - (alpha/(m)) * ((compute_sigmoid(x_norm*theta)-y)' * x_norm)';
        % Save the cost J in every iteration so we can later track how it is converging
        J(idx) = compute_cost_logistic_regression(theta,x_norm,y);
    end
    
end