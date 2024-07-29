### 
using Plots, Random
###
function initializeParameters(X, Y)
	positiveYs = [Y[i] >= 1e-5 for i = 1:length(X)];
	numPositives = sum(positiveYs);

	A = [ones(numPositives) X[positiveYs]];

	l, initialParamB = A \ log.(Y[positiveYs]);

	return exp(l), initialParamB;
end
###
function GaussNewton(X, Y, maxIter = 100, batchSize = -1)

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

	
	print("Final guess for a after $maxIter iterations with Gauss-Newton: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations with Gauss-Newton: ");
	println(paramB)
	
	return paramA * normVal, paramB, loss;
end
###
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
	print("Final guess for a after $maxIter iterations with Levenberg Marquardt: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations with Levenberg Marquardt: ");
	println(paramB)
	
	return (paramA * normVal, paramB, loss);
end
###
function GradientDescent(X, Y, maxIter = 500, lr = 0.05, batchSize = -1)

	N = length(X);
	println(N);
	
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

	print("Final guess for a after $maxIter iterations with Gradient Descent: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations with Gradient Descent: ");
	println(paramB)
	
	return paramA * normVal, paramB, loss;
	
end
###
begin 

	Points = 100
	noise = 10
	f(x) = π * exp(-.5 * x)
	interval = (-10,10)
	
	X = [i for i = interval[1]:(interval[2]-interval[1])/Points:interval[2]]
	Y = [f(x) + (rand()-0.5) * noise for x in X]

	# Plots our data
	scatter(X, Y)
	plot!(f)
 
end
###
begin
	a,b,loss = GaussNewton(X,Y,100)
	println()
	a2,b2,loss2 = GradientDescent(X,Y, 999 ,0.005,25)
	println()
	a3,b3,loss3 = LevenbergMarquardt(X,Y)
	
	scatter(X,Y)
	plot!(x->a*exp(b*x),lw=2,ls=:dash,color=:purple,label="Gauss-Newton")
	plot!(x->a2*exp(b2*x),lw=1.5,color=:red,ls=:dash,label="Gradient")
	plot!(x->a3*exp(b3*x),lw=1.2,label="Levenberg Marquardt")
end
###
