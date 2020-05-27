function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis = g(X*theta)
h = sigmoid(X * theta);

% Regularized cost has two components:
%  1) general cost
%  2) regularized cost that penalizes the params.
cost1 = -(y' * log(h) + (1-y)' * log(1-h))/m;

% regularized cost to exclude bias param.
theta(1) = 0;
cost2 = theta'*theta * lambda/(2*m);

% regularized cost
J = cost1 + cost2;

% regularized grad
grad = (X'*(h-y) + lambda*theta)/m;

% =============================================================

end
