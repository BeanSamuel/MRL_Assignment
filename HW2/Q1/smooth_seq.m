% 數據初始化
T = [0.7 0.3; 0.4 0.6];
Z = [0.1 0.9; 0.7 0.3];
measurement = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0];
initial_state = [0.5; 0.5];

%Filter
forward_results = bayes_filter(T, Z, measurement, initial_state);

%Backward
backward_results = bayes_backward(T, Z, measurement);

%Smoothing
smoothed_results = zeros(2, length(measurement));
for t = 1:length(measurement)
    smoothed_estimate = forward_results(:, t) .* backward_results(:, t);
    smoothed_estimate = smoothed_estimate / sum(smoothed_estimate); % Normalize
    smoothed_results(:, t) = smoothed_estimate;
end

% Display results
for t = 1:size(smoothed_results, 2)
    fprintf('Time %d: P(X_t=0) = %.4f, P(X_t=1) = %.4f\n', t, smoothed_results(2, t), smoothed_results(1, t));
end

%Filter Function
function results = bayes_filter(T, Z, measurement, initial_state)
    current_state = initial_state;
    num_measurements = length(measurement);
    results = zeros(2, num_measurements);

    for t = 1:num_measurements
        z = measurement(t);
        current_state = T' * current_state;
        current_state = current_state .* Z(:, z + 1);
        current_state = current_state / sum(current_state);
        results(:, t) = current_state;
    end
end

%Backward Function
function backward_results = bayes_backward(T, Z, measurement)
    num_measurements = length(measurement);
    beta = ones(2, 1);
    backward_results = zeros(2, num_measurements);
    backward_results(:, num_measurements) = beta;

    for t = num_measurements:-1:2
        z_next = measurement(t);
        beta = T' * (Z(:, z_next + 1) .* beta);
        backward_results(:, t-1) = beta;
    end
end
