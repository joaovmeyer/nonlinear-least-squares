function Newton(X, Y, maxIter = 100, batchSize = -1)

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

	# normalizing Y value
	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;


	# tunning the initial parameters
	paramA, paramB = initializeParameters(X, Y_norm);

	loss = [];

	# points seen since last update
	pointsSeen = 0;

	# have these outside the training loop because if we are usign batches, the remainder of one batch can be used in the next one.
	JTJ = zeros(2, 2);
	JTr = zeros(2, 1);

	for iter = 1:maxIter
		L = 0.0;

		for i = 1:N
			e = exp(paramB * X[i]);

			JTJ[1, 1] += e * e;
			JTJ[1, 2] += paramA * X[i] * e * e;
			JTJ[2, 1] += paramA * X[i] * e * e;
			JTJ[2, 2] += paramA * paramA * X[i] * X[i] * e * e;

			ri = paramA * e - Y_norm[i];

			JTr[1, 1] += ri * e;
			JTr[2, 1] += ri * paramA * X[i] * e;
			L += ri * ri;

			pointsSeen += 1;

			if (pointsSeen == batchSize)
				delta = JTJ \ JTr;
				paramA -= delta[1];
				paramB -= delta[2];
			
				JTJ = zeros(2, 2);
				JTr = zeros(2, 1);
				pointsSeen = 0;
			end
		end

		push!(loss, L);
	end

	# maybe at the end some points were seen but not used. It's best to keep
	# it that way, because if we choose to use them, they have a high chance
	# of not representing out function well, due to their small number

	
	print("Final guess for a after $maxIter iterations: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations: ");
	println(paramB)
	
	return paramA * normVal, paramB, loss;
end
