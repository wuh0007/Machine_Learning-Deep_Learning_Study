function [x_norm] = normalize_features(x)
%   find size of the sample = m
m = size(x,1);

% formula implementation x_norm = m x n, x = m x n, mean(x) and std(x)
% need to be m x n, use repmat
x_norm = (x - repmat(mean(x),[m,1])) ./ repmat(std(x),[m,1]);

end

