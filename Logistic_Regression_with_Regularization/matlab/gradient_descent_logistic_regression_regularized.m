function [J,theta] = gradient_descent_logistic_regression_regularized(theta,x_norm,y,alpha,num_iters,lamta)
% gradient descent for logistic regression 
    m = size(x_norm,1);
    
    for idx = 1:num_iters  
        % final theta vector 
        % theta(0)
        theta(1) = theta(1) - (alpha/(m)) * ((compute_sigmoid(x_norm*theta)-y)' * x_norm(:,1))';
        % theta(1-j)
        theta(2:end) = (theta(2:end) .* (1-((alpha*lamta)/m)))  - ((alpha/(m)) * ((compute_sigmoid(x_norm*theta)-y)' * x_norm(:,2:end))');
        % Save the cost J in every iteration so we can later track how it is converging
        J(idx) = compute_cost_logistic_regression_regularized(theta,x_norm,y,lamta);
    end
    
end