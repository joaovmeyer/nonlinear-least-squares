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



	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;
	scatter(X, Y_norm);


	# initialize a and b value
	positiveYs = [Y_norm[i] >= 1e-5 for i = 1:N];
	numPositives = sum(positiveYs);

	A = [ones(numPositives) X[positiveYs]];

	l, initialParamB = A \ log.(Y_norm[positiveYs]);
	initialParamA = exp(l);
	
	print("Initial guess for a: ");
	println(initialParamA * normVal);

	
	print("Initial guess for b: ");
	println(initialParamB)





	# paramA and paramB -> gradient descent
		paramA = initialParamA;
		paramB = initialParamB; 

		# paramA2 and paramB2 -> Newton's method
		paramA2 = initialParamA;
		paramB2 = initialParamB;
	
		lr = 0.001;
		batchSize = 75;
	
		println(N);
	
		iterações = [];
		loss = [];
		loss2 = [];
		
		for iter = 1:1000
			global paramA, paramA2, paramB, paramB2;
		
			c = 0.0;
			d = 0.0;
			gradB = 0.0;
			
			c2 = 0.0;
			d2 = 0.0;
			gradB2 = 0.0;
	
			L = 0.0;
			L2 = 0.0;
			E = 0.0;
		
			for i = 1:N
				x = X[i];
				y = Y_norm[i];
				
				e = exp(paramB * x)
				e2 = exp(paramB2 * x)
				
				c += e * e;
				d += y * e;
				
				c2 += e2 * e2;
				d2 += y * e2;
		
				gradB += (paramA * e - y) * paramA * x * e;
				gradB2 += (paramA2 * e2 - y) * paramA2 * x * e2;
				
				L += (paramA * e - y)^2;
				
				L2 += (paramA2 * e2 - y)^2;
				E += (paramA2 * e2 - y)^2;
		
				if (i % batchSize == 0)
					paramA = d / c;
					paramB -= (gradB / batchSize) * lr;
					
					paramA2 = d2 / c2;
					paramB2 += E / gradB2;
					
					gradB = 0.0
					gradB2 = 0.0;
					c = 0.0;
					d = 0.0;
					c2 = 0.0;
					d2 = 0.0;
					E = 0.0;
				end
	
			end
	
			append!(loss, L);
			append!(loss2, L2);
			append!(iterações, iter);
	
			if (N % batchSize != 0)
				paramA = d / c;
				paramB -= (gradB / (N % batchSize)) * lr;
				
				paramA2 = d2 / c2;
				paramB2 += E / gradB2;
		end
		
		end
		
		paramA *= normVal;
		paramA2 *= normVal;
	
		println("Gradient descent: ", paramA, ", ", paramB);
		println("Newton: ", paramA2, ", ", paramB2);
	
		plot(iterações, loss, label="gradient descent");
		plot!(iterações, loss2, label="Newton");
		plot!(xlabel="iterações", ylabel="precisão")




	function g(x)
		return paramA * exp(paramB * x);
	end
	function g2(x)
		return paramA2 * exp(paramB2 * x);
	end
	
	scatter(X, Y, label="data")
	plot!(g, label="Gradient descent");
	plot!(g2, label="Newton");

end
