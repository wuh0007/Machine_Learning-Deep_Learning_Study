function [J] = compute_cost_mean_square_multi(theta,x_norm,y)
    
    m = size(x_norm,1);
    % h = x*theta which has size of m x 1.
    h = x_norm * theta;
    % Computation
    J = sum((h-y).^2)/(2*m);

end

