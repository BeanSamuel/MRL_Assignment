clc;
clear;

r = 10;
degree = 0.174;
times = 180;
T = 0.2; 

R = [T^2/2, 0; T, 0; 0, T^2/2; 0, T];
R = R * [1, 0; 0, 1] * R';
Q = [0.1,0;0,0.1];

A = [1, T, 0, 0; 0, 1, 0, 0; 0, 0, 1, T; 0, 0, 0, 1];
C = [1, 0, 0, 0; 0, 0, 1, 0];

timestep = (0:times) * T;
z = [r * cos(degree*timestep); r * sin(degree*timestep)] +  0.25 * randn(2, times+1);
estimates = zeros(4, times+1);

x = [r; 0; 0; degree*r];
estimate_sigma = eye(4)*0.001;

for i = 1:times
    estimates(:, i) = x;
    x = A * x;
    estimate_sigma = A * estimate_sigma * A' + R;
    K = estimate_sigma * C' / (C * estimate_sigma * C' + Q);
    x = x + K * (z(:,i) - C * x);
    estimate_sigma = (eye(size(K * C, 1)) - K * C) * estimate_sigma;
end
estimates(:, end) = x;

figure;
hold on;
plot(r * cos(linspace(0, 2*pi)), r * sin(linspace(0, 2*pi)), 'g', 'LineWidth', 3);
plot(z(1, :), z(2, :), 'b', 'LineWidth', 1); 
plot(estimates(1, :), estimates(3, :), 'r', 'LineWidth', 1);
legend('Real', 'Real with noises', 'Estimated');
axis equal;
xlim([-11, 11]);
ylim([-11, 11]);


