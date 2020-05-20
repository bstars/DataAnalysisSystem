A = importdata('testSet.txt')
X = A(:,1:2)
y = A(:, 3)



model = svmTrain(X, y, 1, @(x1, x2) x1' * x2); 


supportvec_idx = find(model.alphas > 1e-7);
w = model.w;
b = model.b
xmin = min(X(:,1));
xmax = max(X(:,1));

x1s = linspace(xmin, xmax, 50);
x2s = - w(1) / w(2) * x1s - model.b / w(2);

w
b



% plot(x1s, x2s, 'b--'); hold on;
% plot(X(:,1), X(:,2), 'ro'); hold on;
% plot(X(supportvec_idx, 1), X(supportvec_idx,2),'b+'); hold off;