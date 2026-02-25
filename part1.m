function [b, val, grad] = sgd(n, f, schedule, epochs, batch_size)
    % b = parameter vector
    % val = function value list
    % grad = gradient value list
    
    % Initialize b.
    b = 0;
    % Randomly draw the datapoints of batch size.
    idx = randperm(n);
    for i = 1:batch_size:n
        batch = idx(i : min(i + batch_size - 1, n));
    end
    
    l = gradient(f, x)
    val(end+1) = f(b);
    grad(end+1) = l(b);


end