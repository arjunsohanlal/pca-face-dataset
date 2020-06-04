function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Initializations
[m, n] = size(X);
U = zeros(n);
S = zeros(n);

% Computing covariance matrix
covariance = (1/m) * X' * X;

% Computing eigenvectors and eigenvalues of the covariance matrix
[U, S, V] = svd(covariance);

end
