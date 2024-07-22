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

	# Plots our data
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
	value_of_normalize = sum(Y) * 0.1
	Y_norm = normalize_vector(Y,value_of_normalize) 

  	# Compare the diference between normalized data and normal data
	scatter(X,Y_norm,color=:red,label="Pontos normalizados",xlim=(-4,1),ylim=(-2,2))
	scatter!(X,Y,label="Pontos sem normalizar")
end
###

###

# Gets an array of true or false where abs(element) < ϵ
function isCloseToZero(vector,ϵ = 0.2)
	return Bool[abs(vector[i]) < ϵ for i = 1:length(vector)]
end
###

###
function initializeParameters(X,Y)
	closeToZero = Y[isCloseToZero(X)]
	
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

	return (initialParamA,initialParamB)
end	
###

###
begin
		
	(initialParamA, initialParamB) = initializeParameters(X,Y_norm)
	
	print("Initial guess for a: ")
	println(initialParamA * value_of_normalize)

	
	print("Initial guess for b: ")
	println(initialParamB)
end
###

###
	function evaluateParam(X,Y,correction_value=1,maxIterations=200,lr=0.05,batchSize=15) 
	# correction_value is used because most of the tests
	# were done using normalized data
	new_Y = copy(Y)
	(a,b) = initializeParameters(X,new_Y)
	

	iterations = []
	loss = []
	
	N = length(X)
	
	for iter = 1:maxIterations
		println((a,b))
		c = 0.0
		d = 0.0
		gradB = 0.0

		L = 0.0
	
		for i = 1:N
			x = X[i]
			y = new_Y[i]
			
			e = exp(b * x)

			# Sum to get the exact value of a based on b
			# using the partial derivate of dL/da
			c += e * e
			d += y * e
			
			gradB += (a * e - y) * a * x * e
			
			if (i % batchSize == 0)
				a = d / c
					
				b -= (gradB / batchSize) * lr
				
				gradB = 0.0
				c = 0.0
				d = 0.0
			end

			L += (a * e - y)^2
		end

		append!(loss, L)
		append!(iterations, iter)
		
		if (N % batchSize != 0 && d != 0)
			a = d / c
			b -= (gradB / batchSize) * lr
		end

	end
	
		return (a * correction_value, b, iterations, loss)
	end
###

###
begin
	function g(x)
		return paramA * exp(paramB * x)
	end
	
	scatter(X, Y)
	plot!(g)
end
###

###

function compareResults(a::Float64,b::Float64,X::Vector,noise::Float64,normalize=false::Bool)
	f(x) = a * exp(b * x)
	Y = [f(i) + (rand()-.5) * noise for i=1:length(X)]
	correction_value = sum(Y) * .1
	
	if normalize
		
		Y = normalize_vector(Y,correction_value)
		
		paramA,paramB,iterations,loss = evaluateParam(X,Y,correction_value)
		
	else
	
		Y = normalize_vector(Y,sign(correction_value))
		println(Y)
		paramA,paramB,iterations,loss = evaluateParam(X,Y,correction_value)
		
	end
	
	return (a,b,paramA,paramB,iterations,loss)
end

###
	
###
begin

	local realA,realB,a,b,iterações,loss = compareResults(-.5,1.2,[i for i =-3:0.2:5],10.,true)
	
	plot(x->paramA * exp(paramB * x),label="Função Aproximada")
	plot(x->realA * exp(realB * x),label="Função Real",lw=2)
	scatter!(X,Y,label="Pontos com ruído")

end
###
