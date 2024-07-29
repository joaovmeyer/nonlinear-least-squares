function LevenbergMarquardt(X, Y, maxIter = 100, damping = 5, batchSize = -1)

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
	pointsSeen = 0;
	
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
				
		    	# delayed gratification
				if (iter != 1 && L > loss[end])
					damping *= 2;
				else
					damping /= 3;
				end
		
				JTJ[1, 1] += damping * JTJ[1, 1];
				JTJ[2, 2] += damping * JTJ[2, 2];

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
	print("Final guess for a after $maxIter iterations: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations: ");
	println(paramB)
	
	return (paramA * normVal, paramB, loss);
end
