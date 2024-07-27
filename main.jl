###
begin
	using Plots
	using Random

###

###
function Newton(X, Y, maxIter = 100,initialParameters=true)

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
		
		# a = epx(ln(a))
		initialParamA = exp(l);
		
		
		print("Initial guess for a with Newton: ");
		println(initialParamA * normVal);
	
		
		print("Initial guess for b with Newton: ");
		println(initialParamB)
	
		# tunning the initial parameters
	
		paramA = initialParamA;
		paramB = initialParamB;
	
	else
		
		paramA,paramB = 0.1, 0.1
		
	end


	loss = [];
	
	for iter = 1:maxIter
		
		loss_ri = 0;
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
			loss_ri += ri * ri
			
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
function LevenbergMarquardt(X, Y, maxIter = 100,damping=3,initialParameters=true)

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
		
		# a = epx(ln(a))
		initialParamA = exp(l);
		
		
		print("Initial guess for a with LevenbergMarquardt: ");
		println(initialParamA * normVal);
	
		
		print("Initial guess for b with LevenbergMarquardt: ");
		println(initialParamB)
	
		# tunning the initial parameters
		paramA = initialParamA;
		paramB = initialParamB;

	else
		paramA,paramB = 0.1, 0.1;		
	end


	loss = [];
	
	for iter = 1:maxIter
		
		loss_ri = 0;
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
			loss_ri += ri * ri
			
		end
		append!(loss,loss_ri)
		
		# delayed gratification
		if (iter != 1 && loss_ri > loss[end])
			damping *= 2
		else
			damping /= 3
		end

		JTJ[1,1] += damping * JTJ[1,1]
		JTJ[2,2] += damping * JTJ[2,2]
		
		
		delta = JTJ \ JTr;
		paramA -= delta[1];
		paramB -= delta[2];
		
	end
	print("Final guess for a after $maxIter iterations with LevenbergMarquardt: ");
	println(paramA * normVal);

	
	print("Final guess for b after $maxIter iterations with LevenbergMarquardt: ");
	println(paramB)
	
	return (paramA * normVal, paramB,loss);
end
###

###
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
	a2,b2,loss2 = GradientDescent(X,Y, 500 ,0.05, 25)
	a3,b3,loss3 = LevenbergMarquardt(X,Y)
	
	scatter(X,Y)
	plot!(x->a*exp(b*x),lw=2,ls=:dash,color=:purple,label="Newton")
	plot!(x->a2*exp(b2*x),lw=1.5,color=:red,ls=:dash,label="Gradient")
	plot!(x->a3*exp(b3*x),lw=1.2,label="LevenbergMarquardt")
end
###

###
begin
	mini = minimum(vec([length(loss) length(loss2) length(loss3)]))
		
	interval = [i for i = 1:mini]
	
	plot(title="Comparação Erros",ylims=(0,10))
	plot!(interval,[loss[i] for i = 1:mini],label="Newton")
	plot!(interval,[loss2[i] for i = 1:mini],label="Gradient")
	plot!(interval,[loss3[i] for i = 1:mini],label="Levenberg")
end
###
