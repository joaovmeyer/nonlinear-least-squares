function GradientDescent(X,Y,maxIterations=500,lr=0.05,batchSize=50,initialParameters=true)
	
	# normalizing Y value
	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;

	######### initializes a and b parameters #########
	N = length(X);

	# normalizing Y value
	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;

	if initialParameters
		# initialize a and b value
		positiveYs = [Y_norm[i] >= 1e-5 for i = 1:N];
		numPositives = sum(positiveYs);
	
		A = [ones(numPositives) X[positiveYs]];
	
		# solves the linear system for initial parameters a and b
		l, initialParamB = A \ log.(Y_norm[positiveYs]);
		
		# a = exp(ln(a))
		initialParamA = exp(l);
		
		
		print("Initial guess for a with GradientDescent: ");
		println(initialParamA * normVal);
	
		
		print("Initial guess for b with GradientDescent: ");
		println(initialParamB)
		
		
		paramA = initialParamA;
		paramB = initialParamB;
	
	else
		paramA,paramB = 0.0, 0.0		
	end
	
	##################################################
	
		### Training the parameters ###
	
	loss = []
	
	for iter = 1:maxIterations
		
		c = 0.0
		d = 0.0
		gradB = 0.0

		L = 0.0
	
		for i = 1:N
			x = X[i]
			y = Y_norm[i]
			
			e = exp(paramB * x)

			# Sum to get the exact value of a based on b
			# using the partial derivate of dL/da
			c += e * e
			d += y * e
			
			
			gradB += (paramA * e - y) * paramA * x * e
			L += (paramA * e - y)^2


			#update the parameters every {batchSize} iterations
			if (i % batchSize == 0)		
				
				paramA = d / c
					
				paramB -= (gradB / batchSize) * lr
				
				gradB = 0.0
				c = 0.0
				d = 0.0
			end

		end

		append!(loss, L)

	end
	
	print("Final guess for a after $maxIterations iterations with GradientDescent: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIterations iterations with GradientDescent: ");
	println(paramB)
	
	return (paramA * normVal,paramB,loss)
end	
