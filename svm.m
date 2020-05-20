A = importdata('testSet.txt')
X = A(:,1:2)
y = A(:, 3)

model = svmTrain(X, y, 1, @(x1, x2) x1' * x2); 
plot(X(:,1), X(:,2))

supportvec_idx = find(model.alphas > 0)