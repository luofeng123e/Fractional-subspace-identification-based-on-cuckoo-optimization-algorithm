tic
clc;
close all;
clear all;
%% 对迭代次数q[4]、记忆步长L[50,300]、分数阶阶次gam[0,2]、采样步长h[0,0.1]、滤波参数p[0,0.1]、b[1,10]、初始状态值x0[0,100]
N=50;
D=8;
T=5;
X=[1,50 ,0.001,0.001,0.001,1 ,0.001,0.001;
   6,300,2    ,0.1  ,0.1  ,10,200  ,200   ];
beta = 1.5 ;%levy飞行参数
alpha_max=1;
alpha_min=0.2;
Pa_max=0.7;%随机游走参数
Pa_min=0.1;
F0=0.7;     %变异率
CR=0.2;     %交叉概率
pop=zeros(N,D);
for j=1:D
    pop(:,j)=X(1,j)+rand(N,1).*(X(2,j)-X(1,j));
end
L=randperm(N,round(N/2));
E=setdiff(randperm(N),L);
L_pop=pop(L,:); %布谷鸟种群
E_pop=pop(E,:); %差分进化种群
N=N/2;
rand_pop=zeros(N,D);
M_pop=zeros(N,D);    %变异种群
C_pop=zeros(N,D);    %选择种群
L_pbest=zeros(N,1); %布谷鸟种群适应度值
E_pbest=zeros(N,1); %差分进化种群适应度值
DE_CS_gbest=zeros(T,1);
best_pop=zeros(T,D);
for i=1:N
    L_pbest(i)=func_fit_bian(L_pop(i,:));
    E_pbest(i)=func_fit_bian(E_pop(i,:));
end
for t=1:T
    %% 布谷鸟算法更新种群 %%
    %% levy飞行更新种群位置%%
    sigma_u = (gamma(1+beta)*sin(pi*beta/2)/(beta*gamma((1+beta)/2)*2^((beta-1)/2)))^(1/beta) ;
    sigma_v = 1 ;
    u = normrnd(0,sigma_u,N,D) ;%(第一个参数代表均值，sigma参数代表标准差),生成N×D形式的正态分布的随机数矩阵。
    v = normrnd(0,sigma_v,N,D) ;
    step = u./(abs(v).^(1/beta)) ;
    alpha=alpha_max*exp((1/T)*log(alpha_min/alpha_max)); %自适应步长因子
    levy_pop = L_pop+alpha.*step; %levy飞行更新
    % 边界处理
    for j=1:D
        for i=1:N
            if levy_pop(i,j)<X(1,j)
                levy_pop(i,j)=X(1,j);
            elseif levy_pop(i,j)>X(2,j)
                levy_pop(i,j)=X(2,j);
            end
        end
    end
    %% 以更优的粒子去代替以前粒子，比较适应度值%%
    for i=1:N
        if func_fit_bian(levy_pop(i,:))<func_fit_bian(L_pop(i,:));
            L_pop(i,:)=levy_pop(i,:);
        end
    end
    %%  随机游走更新粒子位置%%
    for i=1:N
        L_pbest(i)=func_fit_bian(L_pop(i,:));
    end
    [~,index]=min(L_pbest);
    Pa=0.2*exp(t/T)*rand; %自适应发现概率
    for i=1:N
        rand_pop(i,:) = L_pop(i,:)+rand.*heaviside(rand-Pa).*(L_pop(index,:)-L_pop(randi(N),:));
    end
    % 边界处理
    for j=1:D
        for i=1:N
            if rand_pop(i,j)<X(1,j)
                rand_pop(i,j)=X(1,j);
            elseif rand_pop(i,j)>X(2,j)
                rand_pop(i,j)=X(2,j);
            end
        end
    end
    %% 以更优的粒子去代替以前粒子，比较适应度值%%
    for i=1:N
        if func_fit_bian(rand_pop(i,:))<func_fit_bian(L_pop(i,:));
            L_pop(i,:)=rand_pop(i,:);
        end
    end
    % 计算更新之后的布谷鸟钟群局部最优适应度值
    for i=1:N
        E_pbest(i)=func_fit_bian(E_pop(i,:));
    end 
    %% 差分进化算法更新种群 %%
    %% 变异操作
    for i=1:N
        r1=randi([1,N],1,1);
        while(r1==i)
            r1=randi([1,N],1,1);
        end
        r2=randi([1,N],1,1);
        while((r2==r1)||(r1==i))
            r2=randi([1,N],1,1);
        end
        r3=randi([1,N],1,1);
        while((r3==r2)||(r2==r1)||(r1==i))
            r3=randi([1,N],1,1);
        end
        M_pop(i,:)=E_pop(r1,:)+F0*(E_pop(r2,:)-E_pop(r3,:));
    end
    % 边界处理
    for j=1:D
        for i=1:N
            if M_pop(i,j)<X(1,j)
                M_pop(i,j)=X(1,j);
            elseif M_pop(i,j)>X(2,j)
                M_pop(i,j)=X(2,j);
            end
        end
    end
    %% 以更优的粒子去代替以前粒子，比较适应度值%%
    for i=1:N
        if func_fit_bian(M_pop(i,:))<func_fit_bian(E_pop(i,:));
            E_pop(i,:)=M_pop(i,:);
        end
    end
    %% 交叉操作
    r=randi([1,D],1,1);
    for j=1:D
        cr=rand;
        if (cr<CR)||(r==j)
            C_pop(:,j)=M_pop(:,j);
        else
            C_pop(:,j)=E_pop(:,j);
        end
    end
    %% 边界处理
    for j=1:D
        for i=1:N
            if C_pop(i,j)<X(1,j)
                C_pop(i,j)=X(1,j);
            elseif C_pop(i,j)>X(2,j)
                C_pop(i,j)=X(2,j);
            end
        end
    end
    %% 选择操作  以更优的粒子去代替以前粒子，比较适应度值
    for i=1:N
        if func_fit_bian(C_pop(i,:))<func_fit_bian(E_pop(i,:));
            E_pop(i,:)=C_pop(i,:);
        end
    end
    % 计算更新之后的布谷鸟钟群局部最优适应度值
    for i=1:N
        E_pbest(i)=func_fit_bian(E_pop(i,:));
    end
    %% 双种群交换最优值 %%
    [~,L_index]=sort(L_pbest,'descend');
    [~,E_index]=sort(E_pbest,'descend');
    for j=1:2 %将各种群中适应度值最好的j个粒子替换到另外种群中适应度值最差的j个粒子上
        L_pop(L_index(j),:)=E_pop(E_index(N+1-j),:);
        E_pop(E_index(j),:)=L_pop(L_index(N+1-j),:);
    end
    %% 记录寻优曲线
    pop=[L_pop;E_pop];
    pbest=[L_pbest;E_pbest];
    [~,index]=min(pbest);
    DE_CS_gbest(t)=pbest(index); %记录寻优曲线
    best_pop(t,:)=pop(index,:); %记录每一代中适应度值最好的辨识参数
end
[A,B,C,D]=func_bian(best_pop(T,:)); %记录全局最优的系统矩阵
figure(1)
plot(DE_CS_gbest);
title(['DECS','最小值',num2str(DE_CS_gbest(T))]);

toc