function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% Initializations
X_rec = zeros(size(Z, 1), size(U, 1));

% Computing the approximation of the data by projecting back
% onto the original space using the top K eigenvectors in U.

U_reduce = U(:,1:K);
X_rec = Z * U_reduce';

end
