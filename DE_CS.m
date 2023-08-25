tic
clc;
close all;
clear all;
%% �Ե�������q[4]�����䲽��L[50,300]�������׽״�gam[0,2]����������h[0,0.1]���˲�����p[0,0.1]��b[1,10]����ʼ״ֵ̬x0[0,100]
N=50;
D=8;
T=5;
X=[1,50 ,0.001,0.001,0.001,1 ,0.001,0.001;
   6,300,2    ,0.1  ,0.1  ,10,200  ,200   ];
beta = 1.5 ;%levy���в���
alpha_max=1;
alpha_min=0.2;
Pa_max=0.7;%������߲���
Pa_min=0.1;
F0=0.7;     %������
CR=0.2;     %�������
pop=zeros(N,D);
for j=1:D
    pop(:,j)=X(1,j)+rand(N,1).*(X(2,j)-X(1,j));
end
L=randperm(N,round(N/2));
E=setdiff(randperm(N),L);
L_pop=pop(L,:); %��������Ⱥ
E_pop=pop(E,:); %��ֽ�����Ⱥ
N=N/2;
rand_pop=zeros(N,D);
M_pop=zeros(N,D);    %������Ⱥ
C_pop=zeros(N,D);    %ѡ����Ⱥ
L_pbest=zeros(N,1); %��������Ⱥ��Ӧ��ֵ
E_pbest=zeros(N,1); %��ֽ�����Ⱥ��Ӧ��ֵ
DE_CS_gbest=zeros(T,1);
best_pop=zeros(T,D);
for i=1:N
    L_pbest(i)=func_fit_bian(L_pop(i,:));
    E_pbest(i)=func_fit_bian(E_pop(i,:));
end
for t=1:T
    %% �������㷨������Ⱥ %%
    %% levy���и�����Ⱥλ��%%
    sigma_u = (gamma(1+beta)*sin(pi*beta/2)/(beta*gamma((1+beta)/2)*2^((beta-1)/2)))^(1/beta) ;
    sigma_v = 1 ;
    u = normrnd(0,sigma_u,N,D) ;%(��һ�����������ֵ��sigma���������׼��),����N��D��ʽ����̬�ֲ������������
    v = normrnd(0,sigma_v,N,D) ;
    step = u./(abs(v).^(1/beta)) ;
    alpha=alpha_max*exp((1/T)*log(alpha_min/alpha_max)); %����Ӧ��������
    levy_pop = L_pop+alpha.*step; %levy���и���
    % �߽紦��
    for j=1:D
        for i=1:N
            if levy_pop(i,j)<X(1,j)
                levy_pop(i,j)=X(1,j);
            elseif levy_pop(i,j)>X(2,j)
                levy_pop(i,j)=X(2,j);
            end
        end
    end
    %% �Ը��ŵ�����ȥ������ǰ���ӣ��Ƚ���Ӧ��ֵ%%
    for i=1:N
        if func_fit_bian(levy_pop(i,:))<func_fit_bian(L_pop(i,:));
            L_pop(i,:)=levy_pop(i,:);
        end
    end
    %%  ������߸�������λ��%%
    for i=1:N
        L_pbest(i)=func_fit_bian(L_pop(i,:));
    end
    [~,index]=min(L_pbest);
    Pa=0.2*exp(t/T)*rand; %����Ӧ���ָ���
    for i=1:N
        rand_pop(i,:) = L_pop(i,:)+rand.*heaviside(rand-Pa).*(L_pop(index,:)-L_pop(randi(N),:));
    end
    % �߽紦��
    for j=1:D
        for i=1:N
            if rand_pop(i,j)<X(1,j)
                rand_pop(i,j)=X(1,j);
            elseif rand_pop(i,j)>X(2,j)
                rand_pop(i,j)=X(2,j);
            end
        end
    end
    %% �Ը��ŵ�����ȥ������ǰ���ӣ��Ƚ���Ӧ��ֵ%%
    for i=1:N
        if func_fit_bian(rand_pop(i,:))<func_fit_bian(L_pop(i,:));
            L_pop(i,:)=rand_pop(i,:);
        end
    end
    % �������֮��Ĳ�������Ⱥ�ֲ�������Ӧ��ֵ
    for i=1:N
        E_pbest(i)=func_fit_bian(E_pop(i,:));
    end 
    %% ��ֽ����㷨������Ⱥ %%
    %% �������
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
    % �߽紦��
    for j=1:D
        for i=1:N
            if M_pop(i,j)<X(1,j)
                M_pop(i,j)=X(1,j);
            elseif M_pop(i,j)>X(2,j)
                M_pop(i,j)=X(2,j);
            end
        end
    end
    %% �Ը��ŵ�����ȥ������ǰ���ӣ��Ƚ���Ӧ��ֵ%%
    for i=1:N
        if func_fit_bian(M_pop(i,:))<func_fit_bian(E_pop(i,:));
            E_pop(i,:)=M_pop(i,:);
        end
    end
    %% �������
    r=randi([1,D],1,1);
    for j=1:D
        cr=rand;
        if (cr<CR)||(r==j)
            C_pop(:,j)=M_pop(:,j);
        else
            C_pop(:,j)=E_pop(:,j);
        end
    end
    %% �߽紦��
    for j=1:D
        for i=1:N
            if C_pop(i,j)<X(1,j)
                C_pop(i,j)=X(1,j);
            elseif C_pop(i,j)>X(2,j)
                C_pop(i,j)=X(2,j);
            end
        end
    end
    %% ѡ�����  �Ը��ŵ�����ȥ������ǰ���ӣ��Ƚ���Ӧ��ֵ
    for i=1:N
        if func_fit_bian(C_pop(i,:))<func_fit_bian(E_pop(i,:));
            E_pop(i,:)=C_pop(i,:);
        end
    end
    % �������֮��Ĳ�������Ⱥ�ֲ�������Ӧ��ֵ
    for i=1:N
        E_pbest(i)=func_fit_bian(E_pop(i,:));
    end
    %% ˫��Ⱥ��������ֵ %%
    [~,L_index]=sort(L_pbest,'descend');
    [~,E_index]=sort(E_pbest,'descend');
    for j=1:2 %������Ⱥ����Ӧ��ֵ��õ�j�������滻��������Ⱥ����Ӧ��ֵ����j��������
        L_pop(L_index(j),:)=E_pop(E_index(N+1-j),:);
        E_pop(E_index(j),:)=L_pop(L_index(N+1-j),:);
    end
    %% ��¼Ѱ������
    pop=[L_pop;E_pop];
    pbest=[L_pbest;E_pbest];
    [~,index]=min(pbest);
    DE_CS_gbest(t)=pbest(index); %��¼Ѱ������
    best_pop(t,:)=pop(index,:); %��¼ÿһ������Ӧ��ֵ��õı�ʶ����
end
[A,B,C,D]=func_bian(best_pop(T,:)); %��¼ȫ�����ŵ�ϵͳ����
figure(1)
plot(DE_CS_gbest);
title(['DECS','��Сֵ',num2str(DE_CS_gbest(T))]);

toc