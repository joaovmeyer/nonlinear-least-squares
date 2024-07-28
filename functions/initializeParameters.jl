function initializeParameters(X, Y)
	positiveYs = [Y[i] >= 1e-5 for i = 1:length(X)];
	numPositives = sum(positiveYs);

	A = [ones(numPositives) X[positiveYs]];

	l, initialParamB = A \ log.(Y[positiveYs]);

	return exp(l), initialParamB;
end
