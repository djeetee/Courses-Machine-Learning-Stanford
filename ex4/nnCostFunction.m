function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% PART I
X = [ones(m,1) X];

z1 = sigmoid(Theta1 * X');
a2 = [ones(1, size(z1, 2)); z1];
a3 = sigmoid(Theta2 * a2);
h = a3;


yK = zeros(num_labels, m);

for i=1:num_labels,
    yK(i,:) = (y==i);
endfor;

J = (sum(sum(-1 * yK .* log(h) - (1 - yK) .* log(1 - h)))) / m;


% grab stuff from ex2
Theta1Reg = Theta1(:,2:size(Theta1,2));
Theta2Reg = Theta2(:,2:size(Theta2,2));

Reg = (lambda/(2 * m)) * (sum(sum(Theta1Reg .^ 2)) + sum(sum(Theta2Reg .^ 2 )));

J = J + Reg;


% PART II

for k = 1:m,
    % Forward propagation on each example
    a1 = X(k,:);
    z2 = Theta1 * a1';

    a2 = sigmoid(z2);
    a2 = [1 ; a2];

    a3 = sigmoid(Theta2 * a2);
    
    for l = 1:num_labels
        yK(l,:) = (y == l);
    end


    % Backwards as per page 9 of ex4.pdf
    % starting with output layer...
    d3 = a3 - yK(:,k);
    
    % Re-add a bais node for z2
    z2 = [1 ; z2];
    d2 = (Theta2' * d3) .* sigmoidGradient(z2);
    
    % Strip out bais node from resulting d2
    d2 = d2(2:end);

    Theta2_grad = (Theta2_grad + d3 * a2');
    Theta1_grad = (Theta1_grad + d2 * a1);

endfor;


% PART III


% Now divide everything (element-wise) by m to return the partial
% derivatives. Note that for regularization these will have to
% removed/commented out.

Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;

% See page 12 of the ex4.pdf

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end