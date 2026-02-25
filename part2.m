load("Lab2.mat")

% Standardize X1.
X1_st = normalize(X1);

[n, d] = size(X1_st);
assert(n == 569 && d == 60);

% Split data.
n_test = 100;
n_train = n - n_test;

X_train = X1_st(1:n_train, :);
y_train = y1(1:n_train);

X_test = X1_st(n_train+1:end, :);
y_test = y1(n_train+1:end);

% Helper functions
my_predict = @(X, x) 2*(X*x >= 0) - 1;   % maps >=0 to +1, else -1
sensitivity = @(yhat, ytrue) sum((yhat == 1) & (ytrue == 1)) / sum(ytrue == 1);
specificity = @(yhat, ytrue) sum((yhat == -1) & (ytrue == -1)) / sum(ytrue == -1);



% ---- Least squares - QR decomposition ----



[Q, R] = qr(X_train, 0);
b_qr = R \ (Q' * y_train);

y_train_hat_qr = my_predict(X_train, b_qr);
y_test_hat_qr = my_predict(X_test, b_qr);

sens_train_qr = sensitivity(y_train_hat_qr, y_train);
sens_test_qr = sensitivity(y_test_hat_qr, y_test);

spec_train_qr = specificity(y_train_hat_qr, y_train);
spec_test_qr = specificity(y_test_hat_qr, y_test);



% ---- Logistic regression ----



batch_size = 32;
step_size = 0.01;
m_rate = 0.9;
num_epochs = 200;
epsilon = 0.001;    % Gradient norm criterion

% Initialize b and gradient norm.
b_prev = zeros(60, 1);
b_lr = zeros(60, 1);
g_norm = 100;

for k = 1:num_epochs

    b_tilde = b_lr + m_rate * (b_lr - b_prev);
    b_prev = b_lr;
    
    % Randomly pick datapoints of batch size.
    idx = randperm(n_train);
    for j = 1:batch_size:n_train
        if g_norm <= epsilon
            break
        end
        batch = idx(j : min(j + batch_size - 1, n_train));
        
        % Compute g(b_tilde).
        g = zeros(60);
        for i = batch
            x_i = X_train(i, :);
            y_i = y_train(i);
            
            l_i = y_i * x_i / (1 + exp(y_i * x_i * b_tilde));
            g = g - transpose(l_i);
        end
    
        g = g / length(batch);
        g_norm = norm(g);
        
        % Update b.
        b_lr = b_tilde - step_size * g;
    end
end

y_train_hat_lr = my_predict(X_train, b_lr);
y_test_hat_lr = my_predict(X_test, b_lr);

sens_train_lr = sensitivity(y_train_hat_lr, y_train);
sens_test_lr = sensitivity(y_test_hat_lr, y_test);

spec_train_lr = specificity(y_train_hat_lr, y_train);
spec_test_lr = specificity(y_test_hat_lr, y_test);



% ---- SVM ----



% Train Linear SVM (baseline).
M_lin = fitcsvm(X_train, y_train, ...
    'KernelFunction','linear', ...
    'Standardize',true, ...
    'BoxConstraint',1);

y_train_hat_m_lin = my_predict(X_train, M_lin.Beta);
y_test_hat_m_lin = my_predict(X_test, M_lin.Beta);

sens_train_m_lin = sensitivity(y_train_hat_m_lin, y_train);
sens_test_m_lin = sensitivity(y_test_hat_m_lin, y_test);

spec_train_m_lin = specificity(y_train_hat_m_lin, y_train);
spec_test_m_lin = specificity(y_test_hat_m_lin, y_test);

% Train RBF (Gaussian) SVM.
% gamma = 1/(2*sigma^2)
% KernelScale = sigma

M_rbf = fitcsvm(X_train, y_train, ...
    'KernelFunction','rbf', ...
    'Standardize',true, ...
    'BoxConstraint',1, ...
    'KernelScale','auto');   % automatic bandwidth selection

y_train_hat_m_rbf = predict(M_rbf, X_train);
y_test_hat_m_rbf = predict(M_rbf, X_test);

sens_train_m_rbf = sensitivity(y_train_hat_m_rbf, y_train);
sens_test_m_rbf = sensitivity(y_test_hat_m_rbf, y_test);

spec_train_m_rbf = specificity(y_train_hat_m_rbf, y_train);
spec_test_m_rbf = specificity(y_test_hat_m_rbf, y_test);

% ---- Performance evaluation ----

fprintf('\nMETHOD            Train Sens   Test Sens   Train Spec   Test Spec\n');
fprintf('---------------------------------------------------------------\n');
fprintf('QR                    %.3f       %.3f      %.3f   %.3f\n', ...
        sens_train_qr, sens_test_qr, spec_train_qr, spec_test_qr);
fprintf('Logistic Regression   %.3f       %.3f      %.3f   %.3f\n', ...
        sens_train_lr, sens_test_lr, spec_train_lr, spec_test_lr);
fprintf('Linear SVM            %.3f       %.3f      %.3f   %.3f\n', ...
        sens_train_m_lin, sens_test_m_lin, spec_train_m_lin, spec_test_m_lin);
fprintf('RBF SVM               %.3f       %.3f      %.3f   %.3f\n', ...
        sens_train_m_rbf, sens_test_m_rbf, spec_train_m_rbf, spec_test_m_rbf);
