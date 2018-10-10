%% Generalized equation for N hopf oscillators

clear; close all; clc;
N=9; %input the number of nodes
S=0.5*N*(N-1); % number of edges/connections
si=complex(rand(1,S),rand(1,S));
%% Arranging weights as a hermition matrix of size NxN 
%% so that the bidirectional weights are conjugate of one another
c=1;
A1=zeros(N);
for i=1:N-1
    A1(i,i+1:N)=si(c:c+length(i+1:N)-1);
    c=c+length(i+1:N);
end
A2=A1';
w=A1+A2;
% w=rand(N,N);
% w=normalize_hopf(w) % a function which normalizes the afferent 
% connections wi/(sum of absolute of all afferent weights from wi)
%% initialization
z=rand(N,1);
z1=z;
mu=1;
omega=2*pi*[1, sqrt(2), sqrt(3), 2, sqrt(5), sqrt(6), sqrt(7), sqrt(8), 3 ]';
if numel(omega)~=N
    error(['enter ',num2str(N),' initial frequencies']);
end
ci=0.5; % coupling factor
T=50;
dt=0.001;
tt=(0:dt:T);
Z=zeros(N,length(tt));
Z1=zeros(N,length(tt));
idx=1;
W=[];
% abs(w)
eta=0.1;
%% Euler iteration
for i=tt
    Z(:,idx)=z;
    Z1(:,idx)=z1;

    zdot=(mu-abs(z).^2).*z+1i*omega.*z+ci*w*z;
    z1dot=(mu-abs(z1).^2).*z1+1i*omega.*z1;
    wdot=eta*(-w+(z*z'));
    z=z+zdot*dt;
    z1=z1+z1dot*dt;
    w=w+wdot*dt;
    w=w - diag(diag(w)); % inorder to remove self connection
%     abs(w)
%     w=normalize_hopf(w);
    W(:,:,idx)=w;
    idx=idx+1;
    
end
%% Plotting
for j=1:N
    figure(1);
    subplot(4,1,j);
    plot(tt,real(Z(j,:))/abs(max(real(Z(j,:)))));
    hold on; plot(tt,real(Z1(j,:))/abs(max(real(Z1(j,:)))),'r'); xlabel('time');
    hold off;
    axis tight
    title(['oscillator with omega = ', num2str(omega(j)/(2*pi))])
    wt=W(j,:,:);
  figure(4);
   subplot(4,1,j);
    plot(tt,squeeze(abs(wt(1,1,:))),'b',tt,squeeze(abs(wt(1,2,:))),'r',tt,squeeze(abs(wt(1,3,:))),'g');
    
%     yh=hilbert(Z(j,:));
%     phase=(angle(Z(j,:)));
%     phase1=angle(Z1(j,:));
%     figure(3);
%     subplot(4,1,j);
%     plot(tt,phase);
%     hold on; plot(tt,phase1,'r');
end


omega=omega/(2*pi);
figure(1);xlabel('time'); ylabel('normalized real(Z)');
legend('with laterals','without laterals');
figure(4);legend(['f=',num2str(omega(1))],[' f=',num2str(omega(2))],[' f=',num2str(omega(3))]);%,[' f=',num2str(omega(4))]);
figure(2);
subplot(211);
plot(tt,real(Z(1,:))/abs(max(real(Z(1,:)))),'b',tt,real(Z(2,:))/abs(max(real(Z(2,:)))),'r',tt,real(Z(3,:))/abs(max(real(Z(3,:)))),'g');%,tt,real(Z(4,:))/abs(max(real(Z(4,:)))),'c');
title('real(Zi) with laterals');
subplot(212)
plot(tt,real(Z1(1,:))/abs(max(real(Z1(1,:)))),'b',tt,real(Z1(2,:))/abs(max(real(Z1(2,:)))),'r',tt,real(Z1(3,:))/abs(max(real(Z1(3,:)))),'g');%,tt,real(Z1(4,:))/abs(max(real(Z1(4,:)))),'c');
title('real(Zi) without laterals')

legend(['f=',num2str(omega(1))],[' f=',num2str(omega(2))],[' f=',num2str(omega(3))]);%,[' f=',num2str(omega(4))]);
xlabel('time');
ABSOLUTE_W=abs(w)