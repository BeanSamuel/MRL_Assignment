function [Seg,Si_n,S_n] = Segment(xy)

x = xy(:,1);
y = xy(:,2);

threshold= 0.1;

S_i = 1; % size of segment
S_n = 1; % number of segment

n0ind = find(x~=0 | y~=0);
n_s = size(n0ind);
n_0 = n_s(1);

Seg(1,1) = n0ind(1);
for i = 2:n_0
    if sqrt( ( x(n0ind(i)) - x(n0ind(i-1)) )^2 + ( y(n0ind(i)) - y(n0ind(i-1)) )^2  ) < threshold
        S_i = S_i + 1;
        Seg(S_i,S_n) = n0ind(i);
    else
        S_n = S_n + 1;
        S_i = 1;
        Seg(S_i,S_n) = n0ind(i);
    end
end

Si_n = zeros(S_n,1);
for j = 1:S_n
    k = size(find(Seg(:,j)~=0));
    Si_n(j) = k(1);
end

end