function [w, history] = sgd_simple(fun, grad, w0, n, stepSchedule, eta0, epochs, batchSize)
%SGD_SIMPLE  Minimal stochastic gradient descent with basic diagnostics.
%
% Inputs
%   fun          : f = fun(w, batchIdx)          (objective on a batch)
%   grad         : g = grad(w, batchIdx)         (gradient on a batch)
%   w0           : initial parameter vector
%   n            : number of data points
%   stepSchedule : 'constant' | 'inv' | 'invsqrt'
%   eta0         : base step size
%   epochs       : number of epochs
%   batchSize    : mini-batch size
%
% Outputs
%   w            : final parameter vector
%   history      : struct with diagnostics per epoch:
%                  history.f(e) = fun value (full data)
%                  history.gn(e)= grad norm (full data)

    w = w0(:);

    history.f  = zeros(epochs,1);
    history.gn = zeros(epochs,1);

    k = 0; % total update counter

    for e = 1:epochs
        idx = randperm(n); % shuffle each epoch

        for i = 1:batchSize:n
            k = k + 1;

            batch = idx(i : min(i+batchSize-1, n));

            % step size
            switch stepSchedule
                case 'constant'
                    eta = eta0;
                case 'inv'      % O(1/k)
                    eta = eta0 / k;
                case 'invsqrt'  % O(1/sqrt(k))
                    eta = eta0 / sqrt(k);
                otherwise
                    error('stepSchedule must be constant, inv, or invsqrt');
            end

            % SGD update
            g = grad(w, batch);
            w = w - eta * g;
        end

        % diagnostics at end of epoch (evaluate on full data)
        allIdx = 1:n;
        history.f(e)  = fun(w, allIdx);
        history.gn(e) = norm(grad(w, allIdx));
    end
end

function [] = plot_history(f_history,grad_history)
    figure;
    semilogy(f_history, 'LineWidth', 1.5);
    hold on;
    semilogy(grad_history, 'LineWidth', 1.5);
    grid on;
    legend('f(x)', '||∇f||');
    xlabel('Epoch');
    ylabel('Log scale value');
    title('Convergence diagnostics');
    hold off;
end 
%% Problem setup

d = 1000;
n = d;


% Random orthonormal matrix Q from QR
[Q, ~] = qr(randn(d));

% Positive diagonal eigenvalues (controls conditioning)
lambda = logspace(0, 3, d)';   % from 1 to 1e3 (all positive)
% lambda = rand(d,1) + 0.1; 
D = diag(lambda);

% Symmetric positive definite matrix
%M = Q * D * Q';

A = diag(sqrt(lambda)) * Q';
M =  A' * A;

%% Gradient Descent

fun  = @(x, batch) 0.5/numel(batch) * norm(A(batch,:)*x)^2;
grad = @(x, batch) (A(batch,:)' * (A(batch,:)*x)) / numel(batch);

x0 = randn(d,1);

[w_gd, hist_gd] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, n);
%plot_history(hist_gd.f, hist_gd.gn);
f_true = 0.5 * (w_gd' * M * w_gd);
fprintf('Final error GD = %.6e\n', f_true);

%% SGD

[w_sgd1, hist_sgd1] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, 1);
%plot_history(hist_sgd1.f, hist_sgd1.gn);
f_true = 0.5 * (w_sgd1' * M * w_sgd1);
fprintf('Final error SGD (MB=1) = %.6e\n', f_true);

[w_sgd2, hist_sgd2] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, 4);
%plot_history(hist_sgd2.f, hist_sgd2.gn);
f_true = 0.5 * (w_sgd2' * M * w_sgd2);
fprintf('Final error SGD (MB=4) = %.6e\n', f_true);

[w_sgd3, hist_sgd3] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, 8);
%plot_history(hist_sgd3.f, hist_sgd3.gn);
f_true = 0.5 * (w_sgd3' * M * w_sgd3);
fprintf('Final error SGD (MB=8) = %.6e\n', f_true);

[w_sgd4, hist_sgd4] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, 16);
%plot_history(hist_sgd4.f, hist_sgd4.gn);
f_true = 0.5 * (w_sgd4' * M * w_sgd4);
fprintf('Final error SGD (MB=16) = %.6e\n', f_true);

[w_sgd5, hist_sgd5] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, 32);
%plot_history(hist_sgd5.f, hist_sgd5.gn);
f_true = 0.5 * (w_sgd5' * M * w_sgd5);
fprintf('Final error SGD (MB=32) = %.6e\n', f_true);

[w_sgd6, hist_sgd6] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, 64);
%plot_history(hist_sgd6.f, hist_sgd6.gn);
f_true = 0.5 * (w_sgd6' * M * w_sgd6);
fprintf('Final error SGD (MB=64) = %.6e\n', f_true);

[w_sgd7, hist_sgd7] = sgd_simple(fun, grad, x0, n, 'constant', 1e-3, 1000, 128);
%plot_history(hist_sgd7.f, hist_sgd7.gn);
f_true = 0.5 * (w_sgd7' * M * w_sgd7);
fprintf('Final error SGD (MB=128) = %.6e\n', f_true);

%% GD and SGD with step-size scheduling

[w_gd_inv, hist_gd_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, n);
%plot_history(hist_gd.f, hist_gd.gn);
f_true = 0.5 * (w_gd_inv' * M * w_gd_inv);
fprintf('Final error GD (inv) = %.6e\n', f_true);

[w_sgd1_inv, hist_sgd1_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, 1);
%plot_history(hist_sgd1.f, hist_sgd1.gn);
f_true = 0.5 * (w_sgd1_inv' * M * w_sgd1_inv);
fprintf('Final error SGD (MB=1) (inv) = %.6e\n', f_true);

[w_sgd2_inv, hist_sgd2_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, 4);
%plot_history(hist_sgd2.f, hist_sgd2.gn);
f_true = 0.5 * (w_sgd2_inv' * M * w_sgd2_inv);
fprintf('Final error SGD (MB=4) = %.6e\n', f_true);

[w_sgd3_inv, hist_sgd3_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, 8);
%plot_history(hist_sgd3.f, hist_sgd3.gn);
f_true = 0.5 * (w_sgd3_inv' * M * w_sgd3_inv);
fprintf('Final error SGD (MB=8) = %.6e\n', f_true);

[w_sgd4_inv, hist_sgd4_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, 16);
%plot_history(hist_sgd4.f, hist_sgd4.gn);
f_true = 0.5 * (w_sgd4_inv' * M * w_sgd4_inv);
fprintf('Final error SGD (MB=16) = %.6e\n', f_true);

[w_sgd5_inv, hist_sgd5_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, 32);
%plot_history(hist_sgd5.f, hist_sgd5.gn);
f_true = 0.5 * (w_sgd5_inv' * M * w_sgd5_inv);
fprintf('Final error SGD (MB=32) = %.6e\n', f_true);

[w_sgd6_inv, hist_sgd6_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, 64);
%plot_history(hist_sgd6.f, hist_sgd6.gn);
f_true = 0.5 * (w_sgd6_inv' * M * w_sgd6_inv);
fprintf('Final error SGD (MB=64) = %.6e\n', f_true);

[w_sgd7_inv, hist_sgd7_inv] = sgd_simple(fun, grad, x0, n, 'inv', 1e-3, 1000, 128);
%plot_history(hist_sgd7.f, hist_sgd7.gn);
f_true = 0.5 * (w_sgd7_inv' * M * w_sgd7_inv);
fprintf('Final error SGD (MB=128) = %.6e\n', f_true);
