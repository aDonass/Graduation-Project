% 初始化参数
N = 256; % 空间网格点数
L = 2*pi; % 空间域长度
x = linspace(0, L, N+1); % 空间网格
x = x(1:end-1); % 由于周期性边界条件，去掉最后一个点
dx = x(2) - x(1); % 空间步长
dt = 0.001; % 时间步长
T = 1; % 总时间
Nt = floor(T/dt); % 时间步数
visc = 0.01; % 粘性系数

% 初始条件（例如，正弦波）
u0 = sin(x);

% 初始化解向量
u = u0;

% 时间迭代
for n = 1:Nt
    % 使用周期性边界条件和中心差分法求解
    u = u - dt/dx * u .* (circshift(u, -1) - circshift(u, 1))/2 + visc*dt/dx^2 * (circshift(u, -1) - 2*u + circshift(u, 1));
end

% 绘制结果
plot(x, u0, 'b-', x, u, 'r--');
legend('Initial Condition', 'Solution at T=1');
xlabel('x');
ylabel('u');
title('Solution of the Burgers Equation');
