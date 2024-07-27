function gradientDescent(X, Y, maxIter = 500, lr = 0.05, batchSize = 75)

	N = length(X);

	println(N);
	
	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;

	# initialize a and b value
	positiveYs = [Y_norm[i] >= 1e-5 for i = 1:N];
	numPositives = sum(positiveYs);

	A = [ones(numPositives) X[positiveYs]];

	l, initialParamB = A \ log.(Y_norm[positiveYs]);
	initialParamA = exp(l);
	
	print("Initial guess for a: ");
	println(initialParamA * normVal);

	
	print("Initial guess for b: ");
	println(initialParamB);


	# training the parameters
	paramA = initialParamA;
	paramB = initialParamB;

	loss = [];

	for iter = 1:maxIter
		
		gradA = 0.0;
		gradB = 0.0;

		L = 0.0;
	
		for i = 1:N
			x = X[i];
			y = Y_norm[i];
			
			e = exp(paramB * x);

			gradA += (paramA * e - y) * e
			gradB += (paramA * e - y) * paramA * x * e;
			
			L += (paramA * e - y)^2;

			# update the parameters every {batchSize} iterations
			if (i % batchSize == 0)
				paramA -= (gradA / batchSize) * lr;
				paramB -= (gradB / batchSize) * lr;

				gradA = 0.0;
				gradB = 0.0;
			end

		end

		append!(loss, L);

		if (N % batchSize != 0 && false)
			paramA = d / c;
			paramB -= (gradB / (N % batchSize)) * lr;
		end
	end

	return paramA * normVal, paramB, loss;
	
end
