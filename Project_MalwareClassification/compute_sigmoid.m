function [g] = compute_sigmoid(z)
%   Compute sigmod function

%   Initialize the g(z)
g = zeros(size(z));

% sigmoid formula 
g = 1.0 ./ ( 1.0 + exp(-z));

end
