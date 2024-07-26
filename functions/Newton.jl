function Newton(X, Y, maxIter = 100)

	N = length(X);

	# normalizing Y value
	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;

	# initialize a and b value
	positiveYs = [Y_norm[i] >= 1e-5 for i = 1:N];
	numPositives = sum(positiveYs);

	A = [ones(numPositives) X[positiveYs]];

	# solves the linear system for initial parameters a and b
	l, initialParamB = A \ log.(Y_norm[positiveYs]);
	
	# a = epx(ln(a))
	initialParamA = exp(l);
	
	
	print("Initial guess for a: ");
	println(initialParamA * normVal);

	
	print("Initial guess for b: ");
	println(initialParamB)


	# tunning the initial parameters
	paramA = initialParamA;
	paramB = initialParamB;

	for iter = 1:maxIter
		JTJ = zeros(2, 2);
		JTr = zeros(2, 1);

		for i = 1:N
			e = exp(paramB * X[i]);

			# d(Rᵢ)/da
			JTJ[1, 1] += e * e;
			JTJ[1, 2] += paramA * X[i] * e * e;
			JTJ[2, 1] += paramA * X[i] * e * e;
			
			# D(Rᵢ)/db
			JTJ[2, 2] += paramA * paramA * X[i] * X[i] * e * e;

			ri = paramA * e - Y_norm[i];

			JTr[1, 1] += ri * e;
			JTr[2, 1] += ri * paramA * X[i] * e;
		end

		delta = JTJ \ JTr;
		paramA -= delta[1];
		paramB -= delta[2];
		
	end
	print("Final guess for a after $maxIter iterations: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations: ");
	println(paramB)
	
	return (paramA * normVal, paramB);
end
  
