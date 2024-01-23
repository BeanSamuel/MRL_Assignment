%%
%  Name: RSL Leg dataset
%  Description: 
%  This code displays the collected laser data from Minibot.
%  The data rate is 1HZ. The metric is meter.
%  If the XY of data is 0, it represents out of laser range. 
%  Data format:
%  xy_data(720 X 2 X 120): 720 scans, xy data, 120 seconds.
%  Seg (m X n): m data index, n segments.
% 
%  $Revision: 1.0 $
%  Authors: Bing-Xian Lu, Yu-Chung Tsai and Kuo-Shih Tseng. 
%  Date: 2019.03.31
%%
clc
clear

xy_data = importdata('xy_data.mat');
size_xy = size(xy_data);
tol_t = size_xy(3);

for sec = 1:tol_t
    clf;
    title([num2str(sec),' sec.']);
    [Seg,Si_n,S_n] = Segment(xy_data(:,:,sec)); % divide segment per second
    fprintf('There are %i segments at %i sec \n ',S_n,sec);
    hold on; 
    %%%plot_lazer_data%%%
    plot(xy_data(:,1,sec),xy_data(:,2,sec),'.');
    %%%%%%%%%%%%%%%%%%%%%
    %%%plot_label_data%%%
    PN = Label(sec,S_n); % load label
    for i=1:S_n
        if PN(i,1) == 1
           plot(xy_data(Seg(1:Si_n(i),i),1,sec),xy_data(Seg(1:Si_n(i),i),2,sec),'o');
           fprintf('%i-th segment is a leg at %i sec \n ',i,sec);
           fprintf('the first index of this leg segment is %i \n ',Seg(1,i) );           
           fprintf('the first point of the leg is (x=%f,y=%f) \n',xy_data(Seg(1,i),1,sec),xy_data(Seg(1,i),2,sec));
        end 
    end
    %%%%%%%%%%%%%%%%%%%%%
    hold off;
    pause(0.1);
end


