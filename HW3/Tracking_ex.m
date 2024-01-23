clc;
clear;

xy_data = importdata('xy_data.mat');
xy_data_shape = size(xy_data);
duration = xy_data_shape(3);

all_x_coordinates = [];
all_y_coordinates = [];

T = 1;

R = [T^2/2, 0; T, 0; 0, T^2/2; 0, T];
R = R * [1, 0; 0, 1] * R';
Q = [0.1, 0; 0, 0.1];

A = [1, T, 0, 0; 0, 1, 0, 0; 0, 0, 1, T; 0, 0, 0, 1];
C = [1, 0, 0, 0; 0, 0, 1, 0];

estimate_sigma = eye(4);

for t = 1:duration
    clf;
    title([num2str(t), ' sec.']);
    [Seg, Si_n, S_n] = Segment(xy_data(:,:,t));
    PN = Label(t, S_n);
    
    hold on;
    plot(xy_data(:,1,t), xy_data(:,2,t), '.');

    leg_segment = [];
    for i = 1:S_n
        if PN(i, 1) == 1
            leg_segment = [leg_segment; xy_data(Seg(1:Si_n(i), i), :, t)];
            plot(xy_data(Seg(1:Si_n(i), i), 1, t), xy_data(Seg(1:Si_n(i), i), 2, t), 'o');
        end
    end

    if t == 1 % initialize
        mean_leg_segment = mean(leg_segment, 1);
        mean_x = mean_leg_segment(1);
        mean_y = mean_leg_segment(2);
        x = [mean_x; 0; mean_y; 0];
    else
        x = A * x;
        estimate_sigma = A * estimate_sigma * A' + R;
        K = estimate_sigma * C' / (C * estimate_sigma * C' + Q);
        x = x + K * (mean(leg_segment, 1)' - C * x);
        estimate_sigma = (eye(size(K * C, 1)) - K * C) * estimate_sigma;
    end

    all_x_coordinates = [all_x_coordinates; x(1)];
    all_y_coordinates = [all_y_coordinates; x(3)];
    plot(all_x_coordinates, all_y_coordinates, '-r','LineWidth',2);

    plot(x(1), x(3), '+r', 'MarkerSize', 20,'LineWidth',2);
    hold off;
    pause(0.15);
end
