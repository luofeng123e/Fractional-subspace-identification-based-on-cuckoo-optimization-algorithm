%% �������ӿռ��ʶ
function [pbest]=func_fit_bian(pop)
% ��������q[2,8]�����䲽��L[50,300]�������׽״�gam[0,2]����������h[0,0.1]���˲�����p[0,0.1]��b[1,10]����ʼ״ֵ̬x0[0,300]
%% ��������ź�
load U1;
load Y1;
U1=U1;
Y1=Y1;
[m,N]=size(U1);
l=size(Y1,1);
%% �����趨
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
t=h:h:h*N; %����ʱ��
%% �����˲�
f=poisson(t,p,b); %�����˲�����
%% ���������΢������
w=zeros(q,N); %Ȩֵ
W1=zeros(q,N); %΢������
W2=zeros(q,N); %΢������
w(:,1)=1;
W1(:,1)=1;
W2(:,1)=1;
for i=2:q
    for j=2:N
        w(i,j)=((1-(gam*(i-1)+1)/(j-1))*w(i,j-1)); %Ȩֵ����
    end
end
for i=2:q
    W1(i,:)=w(i,:)./h^((i-1)*gam); %΢�����ӣ������״�Ϊ(i-1)*gamʱ wL/T^((i-1)*gam)
end
w_w=w(2:q,N-L+1:N);
w_sum=sum(w_w,2);
w_ave=w_sum/L; %��Ȩ�ؼ��䷨
for i=2:q
    for j=N-L+1:N
    w(i,j)=w_ave(i-1); %ʹ�ô��ʼL�������ڵ�Ȩֵ���
    end
end
for i=2:q
    W2(i,:)=w(i,:)./h^((i-1)*gam); %΢�����ӣ������״�Ϊ(i-1)*gamʱ wL/T^((i-1)*gam)
end
%% ���˲���������΢��
if 2*L<N
    H1=zeros(N,2*L);
    for i=1:N
        for j=1:2*L
            if j<i
                H1(i,j)=0;
            else
                H1(i,j)=f(j-i+1); %�����˲���������Ϊ�˺���ʵ�ֶ��˲�������΢�֡�
            end        %��һ����������˵������ʱ�������΢�����ӺͲ���ʱ��������˲�������ˣ��Դ����ơ�
        end
    end
    H2=zeros(N,N-2*L);
    for i=1:N
        for j=1:N-2*L
            if j<i
                H2(i,j)=0;
            else
                H2(i,j)=f(j-i+1); %�����˲���������Ϊ�˺���ʵ�ֶ��˲�������΢�֡�
            end        %��һ����������˵������ʱ�������΢�����ӺͲ���ʱ��������˲�������ˣ��Դ����ơ�
        end
    end
    G1=W1*H1; %���˲���������΢��
    G2=W2*H2;
    G=[G1,G2];
else
    H=zeros(N,N);
    for i=1:N
        for j=1:N
            if j<i
                H(i,j)=0;
            else
                H(i,j)=f(j-i+1); %�����˲���������Ϊ�˺���ʵ�ֶ��˲�������΢�֡�
            end        %��һ����������˵������ʱ�������΢�����ӺͲ���ʱ��������˲�������ˣ��Դ����ơ�
        end
    end
    G=W1*H;
end
%% ����M(U_L,N)��M(Y_L,N)
% �����������Hankel����
Uh=zeros(m*q,N);
Yh=zeros(l*q,N);
for j=1:N       %�������
    for i=1:q
   Uh(2*(i-1)+1:2*i,j)=conv2(G(i,j),U1(:,j));
   Yh(2*(i-1)+1:2*i,j)=conv2(G(i,j),Y1(:,j));
    end
end 
%% QR�ֽ�
[Q,R]=lu([Uh;Yh]');
L=R';
L11=L(1:q*m,1:q*m);
L21=L((q*m+1):end,1:q*m);
L22=L((q*m+1):end,(q*m+1):end);
%% SVD�ֽ�
[U,S,V]=svd(L22);
ss = diag(S);
n=2;
%% ���ϵͳ����
U11=U(:,1:n);
S11=S(1:n,1:n);
T=U11*S11^0.5;
% A��C
C=T(1:l,1:n);
A=T(1:l*(q-1),1:n)\T(l+1:q*l,1:n); 
% B��D
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
   ju2((q*l-n)*(i-1)+1:i*(q*l-n),:)=z;                   %ju�ĵڶ��е�����
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
%% ��ʶ������
x0=[x1,x2];
y=dlsim(A,B,C,D,U1,x0)';
%% ͼ�αȽ�
e1=abs(Y1(1,:)-y(1,:));
e2=abs(Y1(2,:)-y(2,:));
pbest=sum(e1)+sum(e2);
% figure(2)
% plot(y(1,:),'b--');
% hold on;
% plot(Y1(1,:),'r--');
% legend('��ʶ','ʵ��');
% figure(3)
% plot(y(2,:),'b--');
% hold on;
% plot(Y1(2,:),'r--');
% figure(4)
% plot(e1);
% figure(5)
% plot(e2);
toc