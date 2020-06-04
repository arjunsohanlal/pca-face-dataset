function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% Initializations
Z = zeros(size(X, 1), K);

% Computing the projection of the data using only the top K 
% eigenvectors in U (first K columns). 
U_reduce = U(:, 1:K);

% Quick explanation of some basic vector math that I've applied -
% Typically, for a single example 'x' and projection 'z' - 
%					z = U_reduce' * x
% Expand their dimensions for a full dataset with all examples
% and then - 
%					Z' = U_reduce' * X'
%
% which then equates to - 
%					Z = X * U_reduce

Z = X * U_reduce;
end
