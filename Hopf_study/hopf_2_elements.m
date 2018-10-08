%% Customized code for 2 oscillators
clear; close all;clc;
%% Initialization

w=complex(rand(1,2),rand(1,2));
w12=w(1);
w21=w(2);
mu=1;
omega=2*pi*[3 5]; % input the frequencies of oscillators in the bracket in Hz
omega1=omega(1);
omega2=omega(2);
ci=0.2; % coupling factor. Make this 0 to remove coupling
T=50; % end time
dt=0.001;
tt=(0:dt:T);
z1=rand;
z2=rand;
Z1=zeros(1,length(tt));
Z2=zeros(1,length(tt));
y1=z1;
y2=z2;
Y1=Z1;
Y2=Z2;
idx=1;
eta=0.1;
%% Euler iteration
for i=tt
    Z1(idx)=z1;
    Z2(idx)=z2;
    Y1(idx)=y1;
    Y2(idx)=y2;
    %% updating at each dt
    z1dot=(mu-abs(z1)^2)*z1+1i*omega1*z1+ci*w12*z2;
    z2dot=(mu-abs(z2)^2).*z2+1i*omega2*z2+ci*w21*z1;
    w12dot=eta*(-w12+z1*z2');
    w21dot=eta*(-w21+z2*z1');
    z2=z2+z2dot*dt;
    z1=z1+z1dot*dt;
    w12=w12+w12dot*dt;
    w21=w21+w21dot*dt;
    %% without coupling
    y1dot=(mu-abs(y1)^2)*y1+1i*omega1*y1;
    y2dot=(mu-abs(y2)^2).*y2+1i*omega2*y2;
    y2=y2+y2dot*dt;
    y1=y1+y1dot*dt;
 
    %% Saving for plots
    delw1(idx)= abs(w12dot);
    delw2(idx)= abs(w21dot);
    W1(idx)=abs(w12);
    W2(idx)=abs(w21);
    Wp1(idx)=angle(w12);
    Wp2(idx)=angle(w21);
    idx=idx+1;
    
end
%% Plotting

omega1=omega1/(2*pi);
omega2=omega2/(2*pi);
figure; 
plot(tt,real(Z1)/abs(max(real(Z1))));
hold on; plot(tt,real(Z2)/abs(max(real(Z2))),'r'); title('real(Zi)');xlabel('time');legend(['f=',num2str(omega1)],['f=',num2str(omega2)])
figure;
plot(tt,W1,'b',tt,W2,'r');title('abs(Weight)');legend(['f=',num2str(omega1)],['f=',num2str(omega2)]);xlabel('time');
figure;
plot(tt,Wp1,'b',tt,Wp2,'r');title('phase(Weight)');legend(['f=',num2str(omega1)],['f=',num2str(omega2)]);xlabel('time');
figure;
plot(tt,delw1,'b',tt,delw2,'r');title('abs(deltaW)');legend(['f=',num2str(omega1)],['f=',num2str(omega2)]);xlabel('time');
figure()
subplot(211);plot(tt,real(Z1)/abs(max(real(Z1))));
hold on;plot(tt,real(Z2)/abs(max(real(Z2))),'r'); title('real(Zi) with coupling');legend(['f=',num2str(omega1)],['f=',num2str(omega2)]);xlabel('time');
subplot(212);plot(tt,real(Y1)/abs(max(real(Y1))));
hold on;plot(tt,real(Y2)/abs(max(real(Y2))),'r'); title('real(Zi) without coupling');legend(['f=',num2str(omega1)],['f=',num2str(omega2)]);xlabel('time');
