function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a1 = X;
a2 = sigmoid( [ones(m, 1) a1] * Theta1' );
a3 = sigmoid( [ones(m, 1) a2] * Theta2' );
[m, p]= max(a3, [], 2)





% =========================================================================


end
