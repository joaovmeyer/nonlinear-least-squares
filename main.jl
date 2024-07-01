begin

using Plots;

  
function f(x)
	return 2.4 * exp(0.3 * x);
end

X = [i for i =-5:0.5:5];
Y = [f(x)+(rand()-0.5) * 1 for x in X];

plot(X, [f(x) for x in X]);
scatter!(X, Y)



function makeNormalizer(list, minVal = -1.0, maxVal = 1.0)

	lower = (min(list...));
	higher = (max(list...));

	normalizer(x) = (x - lower) / (higher - lower) * (maxVal - minVal) + minVal;
	unnormalizer(x) = (x - minVal) / (maxVal - minVal) * (higher - lower) + lower;

	return (normalizer, unnormalizer);
	
end

(normalizerX, unnormalizerX) = makeNormalizer(X, -1, 1);
(normalizerY, unnormalizerY) = makeNormalizer((Y), 0.0, 1.0);

normalizedX = normalizerX.(X);
normalizedY = normalizerY.(Y);

scatter(normalizedX, normalizedY);


  



paramA = 0.1;
paramB = 2.0;

lr = 0.5;

for iter = 1:100
	global paramA, paramB;

	c = 0.0;
	d = 0.0;
	gradB = 0.0;

	for i = 1:length(X)

		x = normalizedX[i];
		y = normalizedY[i];
		
		e = exp(paramB * x)
		c += e;
		d += y;

		gradB += (paramA * e - y) * paramA * x * e;
	end

	# dL/da x-intersect
	paramA = d / c;
	paramB -= gradB * lr;

end
	
println(paramA, ", ", paramB);


function g(x)
	return unnormalizerY(paramA * exp(paramB * normalizerX(x)));
end

scatter(X, Y)
plot!(g);

end
