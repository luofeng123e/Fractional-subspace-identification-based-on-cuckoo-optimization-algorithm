%% 记录系统矩阵系数
%% 分数阶子空间辨识
function [A,B,C,D]=func_bian(pop)
% 迭代次数q[4]、记忆步长L[50,300]、分数阶阶次gam[0,2]、采样步长h[0,0.1]、滤波参数p[0,0.1]、b[1,10]、初始状态值x0[0,100]
%% 输入输出信号
load U1;
load Y1;
U1=U1;
Y1=Y1;
[m,N]=size(U1);
l=size(Y1,1);
%% 参数设定
q=ceil(pop(1));
if rem(ceil(pop(2)),2)==0
    L=ceil(pop(2));
else
    L=ceil(pop(2))-1;
end
gam=pop(3);
h=pop(4);
p=pop(5);
b=ceil(pop(6));
x1=pop(7);
x2=pop(8);
t=h:h:h*N; %采样时间
%% 数据滤波
f=poisson(t,p,b); %构造滤波函数
%% 构造分数阶微分算子
w=zeros(q,L+(N-L)/2); %权值
w1=zeros(q,N);
W=zeros(q,L+(N-L)/2); %微分算子
w(:,1)=1;
w1(:,1)=1;
W(:,1)=1;
for i=2:q
    for j=2:N
        w1(i,j)=((1-(gam*(i-1)+1)/(j-1))*w1(i,j-1)); %权值计算
    end
end
w(1:q,1:L)=w1(1:q,1:L);
for i=2:q
    for j=1:(N-L)/2
        w(i,L+j)=w1(i,L+2*j); %变步长记忆法权值计算
    end
end
for i=2:q
    W(i,:)=w(i,:)./h^((i-1)*gam); %微分算子，分数阶次为(i-1)*gam时 wL/T^((i-1)*gam)
end
%% 对滤波函数进行微分
H=zeros(L+(N-L)/2,N);
for i=1:L
    for j=1:N
        if j<i
            H(i,j)=0;
        else
            H(i,j)=f(j-i+1); %构造滤波函数矩阵，为了后面实现对滤波函数的微分。
        end        %对一个采样点来说，采样时间最晚的微分算子和采样时间最早的滤波函数相乘，以此类推。
    end
end
a=0;
for i=1:(N-L)/2
    for j=1:(N-L)/2-a
        H(L+i,L-2+2*i+j)=f(j);%构造滤波函数矩阵，为了后面实现对滤波函数的微分。
    end %对一个采样点来说，采样时间最晚的微分算子和采样时间最早的滤波函数相乘，以此类推。        
    a=a+2;
end
G=W*H;
%% 构造M(U_L,N)和M(Y_L,N)
% 构造输入输出Hankel矩阵
Uh=zeros(m*q,N);
Yh=zeros(l*q,N);
for j=1:N       %卷积处理
    for i=1:q
   Uh(2*(i-1)+1:2*i,j)=conv2(G(i,j),U1(:,j));
   Yh(2*(i-1)+1:2*i,j)=conv2(G(i,j),Y1(:,j));
    end
end 
%% QR分解
[Q,R]=lu([Uh;Yh]');
L=R';
L11=L(1:q*m,1:q*m);
L21=L((q*m+1):end,1:q*m);
L22=L((q*m+1):end,(q*m+1):end);
%% SVD分解
[U,S,V]=svd(L22);
ss = diag(S);
n=2;
%% 求解系统矩阵
U11=U(:,1:n);
S11=S(1:n,1:n);
T=U11*S11^0.5;
% A、C
C=T(1:l,1:n);
A=T(1:l*(q-1),1:n)\T(l+1:q*l,1:n); 
% B、D
U22=U(:,n+1:end);
U2T=U22';
M=U2T*L21*inv(L11);
for i=1:q
ju1((q*l-n)*(i-1)+1:i*(q*l-n),:)=U2T(:,l*(i-1)+1:l*i);
end
z=zeros(q*l-n,n);
for i=1:q-1
for j=1+i:q 
    o=U2T(:,l*(j-1)+1:l*j)*C*A^(j-(i+1));
    z=z+o;  
end
   ju2((q*l-n)*(i-1)+1:i*(q*l-n),:)=z;                   %ju的第二列的数据
    z=zeros(q*l-n,n);
end
ju2((q*l-n)*(q-1)+1:q*(q*l-n),1:n)=0;
ju=[ju1,ju2];
%M
for i=1:q
   MT((q*l-n)*(i-1)+1:i*(q*l-n),:)= M(:,m*(i-1)+1:m*i);
end
E=ju\MT;
D=E(1:l,1:m);
B=E(l+1:end,1:m);
%% 辨识输出结果
x0=[x1,x2];
y=dlsim(A,B,C,D,U1,x0)';
%% 图形比较
e1=abs(Y1(1,:)-y(1,:));
e2=abs(Y1(2,:)-y(2,:));
pbest=sum(e1)+sum(e2);
figure(2)
plot(y(1,:),'b--');
hold on;
plot(Y1(1,:),'r--');
legend('辨识','实际');
figure(3)
plot(y(2,:),'b--');
hold on;
plot(Y1(2,:),'r--');
figure(4)
plot(e1);
figure(5)
plot(e2);
end