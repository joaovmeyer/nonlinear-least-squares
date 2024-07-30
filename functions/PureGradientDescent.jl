function GradientDescent(X, Y, maxIter = 500, lr = 0.05, batchSize = -1)

	N = length(X);
	
	# no batching
	if (batchSize <= 0)
		batchSize = N;
	else
		# shuffle points if using batches
		ordem = randperm(N)
		X = X[ordem]
		Y = Y[ordem]
	end
	
	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;


	# training the parameters
	paramA, paramB = initializeParameters(X, Y_norm);

	loss = [];
	
	pointsSeen = 0;
	gradA = 0.0;
	gradB = 0.0;

	for iter = 1:maxIter
		
		L = 0.0;
	
		for i = 1:N
			x = X[i];
			y = Y_norm[i];
			
			e = exp(paramB * x);

			gradA += (paramA * e - y) * e
			gradB += (paramA * e - y) * paramA * x * e;
			
			L += (paramA * e - y)^2;

			pointsSeen += 1;

			# update the parameters every {batchSize} iterations
			if (pointsSeen == batchSize)
				paramA -= (gradA / batchSize) * lr;
				paramB -= (gradB / batchSize) * lr;

				pointsSeen = 0;
				gradA = 0.0;
				gradB = 0.0;
			end

		end

		append!(loss, L);
	end

	return paramA * normVal, paramB, loss;
	
end
