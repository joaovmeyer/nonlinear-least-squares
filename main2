# text between "###" is the content of a cell in Pluto

###
begin
	using Plots
	using Random
end
###

###
function f(x)
	return -.5 * exp(1.2 * x)
end
###

###
begin 

	noise = 10
	
	X = [i for i =-5:0.05:5]
	Y = [f(x) + (rand()-0.5) * noise for x in X]
	
	
	# shuffle our data
	#ordem = randperm(length(X))
	#X = X[ordem]
	#Y = Y[ordem]

 #Plots our data
	scatter(X, Y)
	plot!(f)
 
end
###

###
function normalize_vector(vector,value)
	new_vector = vector ./ value
	return new_vector
end
###

###
begin
	value_of_normalize = sum(abs.(Y)) * 0.1
	Y_norm = normalize_vector(Y,value_of_normalize) 

  #Compare the diference between normalized data and normal data
	scatter(X,Y_norm,color=:red,label="Pontos normalizados",xlim=(-4,1),ylim=(-2,2))
	scatter!(X,Y,label="Pontos sem normalizar")
end
###

###

#Gets an array of true or false where abs(element) < ϵ
function isCloseToZero(vector,ϵ = 0.2)
	return Bool[abs(vector[i]) < ϵ for i = 1:length(vector)]
end
###

###
function initializeParameters(X,Y)
	closeToZero = Y[isCloseToZero(X)]
	
  #If x->0, then a * exp(b*x) -> a
	initialParamA = sum(closeToZero) / length(closeToZero)

  #b = (ln(y) - ln(a)) / x
	initialParamB = 0.0
 
	valid_data = 0
	
	for i = 1:length(X)
		y = Y_norm[i]
  
		if (y <= 0.1 || abs(X[i]) < 0.2) 
			continue
		end

		
		valid_data += 1
		initialParamB += (log(y) - log(initialParamA)) / X[i]
	
	end
	
	initialParamB /= valid_data

	return (initialParamA,initialParamB)
end	
###

###
begin
		
	(initialParamA, initialParamB) = initializeParameters(X,Y_norm)
	
	print("Initial guess for a: ");
	println(initialParamA * sum(abs.(Y)) * 0.1);

	
	print("Initial guess for b: ");
	println(initialParamB)
end
###

###
begin 
	paramA = initialParamA;
	paramB = initialParamB;

	println(paramA);
	println(paramB);

	N = length(X);
	lr = 0.005;
	batchSize = 25;

	iterações = [];
	loss = [];
	
	for iter = 1:100
		global paramA, paramB;
	
		c = 0.0;
		d = 0.0;
		gradB = 0.0;

		L = 0.0;
	
		for i = 1:0
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
	
	paramA *= value_of_normalize ;
	println(paramA, ", ", paramB);

	plot(iterações, loss, xlabel="iterações", ylabel="precisão")
end
###

###
begin
	function g(x)
		return paramA * exp(paramB * x);
	end
	
	scatter(X, Y)
	plot!(g);
end
###
