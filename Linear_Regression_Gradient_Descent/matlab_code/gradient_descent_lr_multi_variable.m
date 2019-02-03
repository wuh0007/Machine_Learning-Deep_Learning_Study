function [J,theta] = gradient_descent_lr_multi_variable(theta,x_norm,y,alpha,num_iters)
    m = size(x_norm,1);
    
    for idx = 1:num_iters  
        % final theta vector size n x 1 
        theta = theta - (alpha/(m)) * (((x_norm * theta)-y)' * x_norm)';
        % Save the cost J in every iteration so we can later track how it is converging
        J(idx) = compute_cost_mean_square_multi(theta,x_norm,y);
    end
    
    

end

