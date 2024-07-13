begin

	using Plots;
	using Random;



	function f(x)
		return 2.4 * exp(0.3 * x);
	end

	X = [i for i =-10:0.1:15];
	Y = [f(x)+(rand()-0.5) * 1 for x in X];

	# shuffle our data
	ordem = randperm(length(X));
	X = X[ordem];
	Y = Y[ordem];
		
	scatter(X, Y)
	plot!(f);



	Y_norm = Y ./ (sum(abs.(Y)) * 0.1);
	scatter(X, Y_norm);


	# initialize a and b value
	closeToZero = Y_norm[Bool[abs(X[i]) < 0.2 for i = 1:length(X)]]
	initialParamA = sum(closeToZero) / length(closeToZero);

	initialParamB = 0.0;
	s = 0;
	for i = 1:length(X)
		y = Y_norm[i];
		if (y <= 0.1 || abs(X[i]) < 0.1) 
			continue;
		end

		s += 1;
		initialParamB += (log(y) - log(initialParamA)) / X[i];
	end
	initialParamB /= s;
	
	print("Initial guess for a: ");
	println(initialParamA * sum(abs.(Y)) * 0.1);

	
	print("Initial guess for b: ");
	println(initialParamB)





	paramA = initialParamA;
	paramB = initialParamB;

	println(paramA);
	println(paramB);

	N = length(X);
	lr = 0.005;
	batchSize = 25;

	println(N);

	iterações = [];
	loss = [];
	
	for iter = 1:500
		global paramA, paramB;
	
		c = 0.0;
		d = 0.0;
		gradB = 0.0;

		L = 0.0;
	
		for i = 1:N
			x = X[i];
			y = Y_norm[i];
			
			e = exp(paramB * x)
			c += e * e;
			d += y * e;
	
			gradB += (paramA * e - y) * paramA * x * e;
	
			if (i % batchSize == 0)
				paramA = d / c;
				paramB -= (gradB / batchSize) * lr;
				
				gradB = 0.0;
				c = 0.0;
				d = 0.0;
			end

			L += (paramA * e - y)^2;
		end

		append!(loss, L);
		append!(iterações, iter);

		if (N % batchSize != 0)
			paramA = d / c;
			paramB -= (gradB / batchSize) * lr;
		end
	
	end
	
	paramA *= sum(abs.(Y)) * 0.1;
	println(paramA, ", ", paramB);

	plot(iterações, loss);
	plot!(xlabel="iterações", ylabel="precisão");




	function g(x)
		return paramA * exp(paramB * x);
	end
	
	scatter(X, Y)
	plot!(g);


end
