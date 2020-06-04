%% Principal Component Analysis - examining principal features that describe faces

%% Initialization
clear ; close all; clc

fprintf('\nLoading face dataset.\n\n');

% Loading Face dataset
load ('facedataset.mat')

%  Display the first 100 faces in the dataset
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%  Running PCA and visualizing the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

%  Normalizing X by subtracting the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Running PCA
[U, S] = pca(X_norm);

%  Visualizing the top 36 eigenvectors found
displayData(U(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


% Dimension Reduction for Faces -
% Projecting images to the eigen space using the top k eigenvectors 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

% Visualization of Faces after PCA Dimension Reduction -
% Projecting images to the eigen space using the top K eigen vectors and visualizing only using those K dimensions
% Comparing to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;

% Another demonstration of the use of PCA -
% ---
% We run K-means on an image to break it down to a 16-color image,
% after which we plot the RGB values in 3D space.
% To better visualize this 3D data, we apply PCA to map it to a 2D plane.

close all; close all; clc

% Loading image
A = double(imread('bird_small.png'));

A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

% Sampling all indexes randomly
sel = floor(rand(16382, 1) * size(X, 1)) + 1;

% Setting up Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

% Visualize the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

% Plot in 2D
figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
fprintf('Program paused. Press enter to continue.\n');
pause;