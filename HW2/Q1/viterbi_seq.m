T = [0.7 0.3; 0.4 0.6];
Z = [0.1 0.9; 0.7 0.3];
measurement = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0];
initial_state = [0.5; 0.5];

most_likely_sequence = viterbi_algo(T, Z, measurement, initial_state);
disp('Most Likely Sequence:');
disp(most_likely_sequence);

function [most_likely_sequence] = viterbi_algo(T, Z, measurement, initial_state)
    num_measurements = length(measurement);
    dp = zeros(2, num_measurements);
    path = zeros(2, num_measurements);

    dp(:, 1) = initial_state .* Z(:, measurement(1) + 1);

    for t = 2:num_measurements
        for j = 1:2
            [max_prob, idx] = max(dp(:, t-1) .* T(:, j));
            dp(j, t) = max_prob * Z(j, measurement(t) + 1);
            path(j, t) = idx;
        end
    end

    [~, most_likely_sequence(num_measurements)] = max(dp(:, num_measurements));
    most_likely_sequence(num_measurements) = most_likely_sequence(num_measurements) - 1; % convert to 0-based indexing
    
    for t = num_measurements-1:-1:1
        if path(most_likely_sequence(t+1)+1, t+1) - 1 == 0
            tmp = 1;
        else
            tmp = 0;
        end 
        most_likely_sequence(t) = tmp;
    end
end