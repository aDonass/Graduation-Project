% 定义空间变量
x = chebfun('x', [0, 1]);

% 定义初始条件为正弦波
init = sin(2*pi*x);

% 定义时间跨度
tspan = [0, 2];

% 定义空间网格大小
s = 1024;

% 定义粘性系数
visc = 0.01;

% 调用burgers1函数求解Burgers方程
u = burgers1(init, tspan, s, visc);

x = chebfun('x', [0, 1]);
plot(u);
title('Burgers Equation Solution');
xlabel('x');
ylabel('u(x,t)');

