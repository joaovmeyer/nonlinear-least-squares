###
begin
	using Plots
	using Random
end
###

###
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
	
	
	print("Initial guess for a with Newton: ");
	println(initialParamA * normVal);

	
	print("Initial guess for b with Newton: ");
	println(initialParamB)


	# tunning the initial parameters
	paramA = initialParamA;
	paramB = initialParamB;

	loss = [];
	loss_ri = 0;
	
	for iter = 1:maxIter
		JTJ = zeros(2, 2);
		JTr = zeros(2, 1);

		for i = 1:N
			e = exp(paramB * X[i]);

			JTJ[1, 1] += e * e;
			JTJ[1, 2] += paramA * X[i] * e * e;
			
			JTJ[2, 1] += paramA * X[i] * e * e;			
			JTJ[2, 2] += paramA * paramA * X[i] * X[i] * e * e;

			ri = paramA * e - Y_norm[i];

			JTr[1, 1] += ri * e;
			JTr[2, 1] += ri * paramA * X[i] * e;
			loss_ri += ri^2
			
		end

		append!(loss,loss_ri)
		
		delta = JTJ \ JTr;
		paramA -= delta[1];
		paramB -= delta[2];
		
	end
	print("Final guess for a after $maxIter iterations with Newton: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations with Newton: ");
	println(paramB)
	
	return (paramA * normVal, paramB,loss);
end
###

###
function GradientDescent(X,Y,maxIterations=500,lr=0.05,batchSize=50)
	
	# normalizing Y value
	normVal = sum(Y) * 0.1;
	Y_norm = Y ./ normVal;

	######### initializes a and b parameters #########
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
	
	
	print("Initial guess for a with GradientDescent: ");
	println(initialParamA * normVal);

	
	print("Initial guess for b with GradientDescent: ");
	println(initialParamB)

	##################################################

	#=
	
	### Just other method to evaluate initial parameters ###
	### But expects values of x ≈ 0 					 ###
	
	closeToZero = Y_norm[Bool[abs(X[i]) < ϵ for i = 1:N]
	
    #If x->0, then a * exp(b*x) -> a
	initialParamA = sum(closeToZero) / length(closeToZero)
	log_of_A = log(initialParamA)
	
    #b = (ln(y) - ln(a)) / x
	initialParamB = 0.0
 
	valid_data = 0
	
	for i = 1:length(X)
		y = Y_norm[i]
  
		if (y <= 0.1 || abs(X[i]) < 0.2) 
			continue
		end

		
		valid_data += 1
		initialParamB += (log(y) - log_of_A) / X[i]
	
	end
	
	initialParamB /= valid_data

	print("Initial guess for a: ")
	println(initialParamA * normVal)

	
	print("Initial guess for b: ")
	println(initialParamB)
	=#

	### Training the parameters ###
	
	paramA = initialParamA;
	paramB = initialParamB;
	
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
###

###
begin 

	f(x) = π * exp(-.5 * x)
	noise = 100
	
	X = [i for i =-15:0.05:10]
	Y = [f(x) + (rand()-0.5) * noise for x in X]
	
	
	# shuffle our data
	ordem = randperm(length(X))
	X = X[ordem]
	Y = Y[ordem]

	# Plots our data
	scatter(X, Y)
	plot!(f)
 
end
###

###
begin
	a,b,loss = Newton(X,Y,100)
	println()
	a_2,b_2,loss_2 = GradientDescent(X,Y, 500 ,0.05,50)
	
	scatter(X,Y)
	plot!(x->a_2*exp(b_2*x),lw=2,ls=:dash,color=:purple)
	plot!(x->a*exp(b*x),lw=1.5,color=:red,ls=:dash)
end
###
