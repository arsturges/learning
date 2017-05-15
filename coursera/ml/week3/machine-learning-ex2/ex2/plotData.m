function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pos_indices = find(y==1);
neg_indices = find(y==0);
plot(X(pos_indices, 1), X(pos_indices, 2), 'rx')
plot(X(neg_indices, 1), X(neg_indices, 2), 'bx')
xlabel('score on exam 1')
ylabel('score on exam 2')






% =========================================================================



hold off;

end
