%% calc frequency
clear;clc; 
fs=2000;
a=importdata('fn-air-F.txt');
b=a.data;
% c=fft_filter(b(:,2),0,4,1/fs);
[~,~,d,e]=amp_pha(b(:,2),fs);
%%
% clc;clear;
D=0.02;
pp=importdata('T-2.5_L-2_0r-1-F.txt');
qq=pp.data;
r=(qq(:,2)-mean(qq(:,2)))*0.005;%/D;
g=sqrt(sum(r.^2)/length(r))*sqrt(2);
s=pp.data(:,1);
scatter(s,r,'k')
xlabel('t/s')
ylabel('y/D')
% set(gca,'xtick',0:60:120,'ytick',-1:1:1);

%%  damping ratio
clc;clear;
t=importdata('decrement in water.DAT');
u=transpose(t(2,:));

for i=1:10
    v(i)=u(i)/u(i+10);
    w(i)=0.1*log(v(i));
    damping(i)=w(i)/sqrt(4*pi*pi+w(i)*w(i));
end

decrement=mean(w);
damping_ratio=mean(damping);


%% calc A
clear;clc;
foldername=dir('E:\Files\experiment data\small k_0.8-8-120\T-0.5D\L-6_201019\L-6_201209\A');
n=size(foldername,1); 
D=0.02;
c=zeros(n-2,1);
for i=3:n
    m=foldername(i).name;
    a=importdata(m);
    b=(a.data(:,2)-mean(a.data(:,2)))*0.005/D;
    s=length(b);
    c(i-2)=sqrt(sum(b.^2)/s)*sqrt(2);
end
f=reshape(c,3,[]);
g=transpose(mean(f,1));

xlsxname=dir('*.xlsx');
p=importdata(xlsxname.name);

q=p.data.Sheet1(2:end,[1 4]);
fn=1.88;  % natural frequency in water
u=q(:,2)/fn/D;
t=[u,g(2:end)];
scatter(t(:,1),t(:,2))

%% calc osc freq

% clear;clc;
% folder='E:\Files\experiment data\small k_0.8-8-120\displacement force with loadcell\T-2.5\L-3.5_210107\F_A';
% cd (folder)
% foldername=dir('*.txt');
% n=size(foldername,1); 
% % D=0.02;
% fs=2000;
% for i=8:6:n
%     x=foldername(i).name;
%     y=importdata(x);
%     [~,~,r,k]=fft_amp(y.data(:,2),fs);
%     subplot(3,8,(i-2)/6);loglog(r,k,'k')
%     xlim([0 100]);
% end

%% calc force frequency
% clear;clc;
% folder='E:\Files\experiment data\small k_0.8-8-120\displacement force with loadcell\T-2.5\L-3.5_210107\F_A';
% cd (folder)
% foldername=dir('*.txt');
% n=size(foldername,1); 
% % D=0.02;
% fs=2000;
% cutf=8;
% for i=7:6:n
%     x=foldername(i).name;
%     y=importdata(x);
%     Fx=fft_filter(y.data(:,2),0,cutf,1/fs);
%     Fy=fft_filter(y.data(:,4),0,cutf,1/fs);
%     [~,~,r,k]=fft_amp(Fx,fs);
%     subplot(3,8,(i-1)/6);loglog(r,k,'k')
%     xlim([0 8]);
% end



%%

% D=0.02;
% L=0.325;
% p=importdata('T-1_L-4.5_1206_F_A.xlsx');
% q=p.data.Sheet1(2:end,[1 4]);
% fn=1.88; % natural frequency in water
% ur=q(:,2)/fn/D;
% CL_mean=(Lift_time_mean(2:end)-Lift_time_mean(1))./(0.5*1000*D*L*q(:,2).^2);
% CL_fluctuate=Lift_fluctuate(2:end)./(0.5*1000*D*L*q(:,2).^2);
% CD_mean=(Drag_time_mean(2:end)-Drag_time_mean(1))./(0.5*1000*D*L*q(:,2).^2);
% CD_fluctuate=Drag_fluctuate(2:end)./(0.5*1000*D*L*q(:,2).^2);
% F=[ur, CL_mean, CL_fluctuate, CD_mean, CD_fluctuate];
% subplot(2,2,1)
% plot(F(:,1),F(:,2))
% legend('CL mean')
% subplot(2,2,2)
% plot(F(:,1),F(:,3))
% legend('CL fluctuate')
% subplot(2,2,3)
% plot(F(:,1),F(:,4))
% legend('CD mean')
% subplot(2,2,4)
% plot(F(:,1),F(:,5))
% legend('CD fluctuate')



%%
save('results.dat','w','-ascii')

%% export data
clear;clc
p=importdata('results_no_spring.xlsx');
z=p.data.FA;
idx=find(isnan(z(1,:)));
z(:,idx)=[];

jj=1:6:49;
i1=jj+1;
i2=jj+2;
i3=jj+3;
i4=jj+4;
i5=jj+5;

[r,c]=size(z);
c=c/3;
A=zeros(r,c);
A(:,1:2:end-1)=z(:,jj);
A(:,2:2:end)=z(:,i1);
% [r,c]=find(isnan(A));
% B=num2cell(A);
% k=length(r);
% for ii=1:k
% B{r(ii),c(ii)}=[];
% end

CL=zeros(r,c);
CL(:,1:2:end-1)=z(:,jj);
CL(:,2:2:end)=z(:,i2);

CL2=zeros(r,c);
CL2(:,1:2:end-1)=z(:,jj);
CL2(:,2:2:end)=z(:,i3);

CD=zeros(r,c);
CD(:,1:2:end-1)=z(:,jj);
CD(:,2:2:end)=z(:,i4);

CD2=zeros(r,c);
CD2(:,1:2:end-1)=z(:,jj);
CD2(:,2:2:end)=z(:,i5);

%%  force and amplitude

clear;clc;
folder='E:\Files\experiment data\small k_0.8-8-120\displacement force with loadcell\no spring\L-6_210110\F_A';
cd (folder)
foldername=dir('*.txt');
n=size(foldername,1); 

D=0.02;
L=0.325;
fn=1.88; % natural frequency in water
c1=0.260953;
c2=0.239946;
fs=2000;
cutf=8;
t=linspace(0,25,50000)';

bb=[];
d=[];
FL_mean=[];
FL_rms=[];
FD_mean=[];
FD_rms=[];
FLL=[];
% ac=[];
F_cal=[];
Phase=[];


for i=2:2:n
    m=foldername(i).name;
    a=importdata(m);
%     b=(a.data(:,2)-mean(a.data(:,2)))*0.005/D;
    aa=fft_filter(a.data(:,2),0,cutf,1/fs);
    b=(aa-mean(aa))*0.005;
    bb=[bb,b];
    s1=length(b);
    c=sqrt(sum((b/D).^2)/s1)*sqrt(2);
    d=[d;c];
    
    dbdt=gradient(b)./gradient(t);
    ddbdt=gradient(dbdt)./gradient(t);
%     ac=[ac,ddbdt];
    FF=1.132*ddbdt+0.427*dbdt+171.68*b;
    F_cal=[F_cal,FF];
    
    h=foldername(i-1).name;
    k=importdata(h);
%     FL=k.data(:,2)/c1;
%     FD=k.data(:,4)/c2;
    FL=fft_filter(k.data(:,2),0,cutf,1/fs)/c1+0.557*ddbdt;
    FLL=[FLL,FL];
%     FL1=FL(5000:45000);
    FD=fft_filter(k.data(:,4),0,cutf,1/fs)/c2;
%     s2=length(FL1);
    s3=length(FD);
    FL_mean=[FL_mean;mean(FL)];
    FL_rms=[FL_rms;sqrt(sum((FL-mean(FL)).^2)/s3)];
    FD_mean=[FD_mean;mean(FD)];
    FD_rms=[FD_rms;sqrt(sum((FD-mean(FD)).^2)/s3)];
    
    [cc,lags]=xcorr(b,FL);
    [~,I] = max(cc(find(lags==0):end));
    lagdiff=360*I/1000;
    if lagdiff >= 360
        lagdiff=lagdiff-360;
    else
        lagdiff=lagdiff;
    end
    
    Phase=[Phase;lagdiff];

    
end


f=reshape(d,3,[]);
g=transpose(mean(f,1));

jj=foldername(1).name;
kk=importdata(jj);
ll=mean(kk.data(:,2));
p=reshape(FL_mean,3,[]);
Lift_time_mean=transpose(mean(p,1));
q=reshape(FL_rms,3,[]);
Lift_fluctuate=transpose(mean(q,1));
r=reshape(FD_mean,3,[]);
Drag_time_mean=transpose(mean(r,1));
s=reshape(FD_rms,3,[]);
Drag_fluctuate=transpose(mean(s,1));
z=reshape(Phase,3,[]);
PhaseLag=transpose(mean(z,1))/180;

xlsxname=dir('*.xlsx');
dd=importdata(xlsxname.name);
v=dd.data.Sheet1(2:end,[1 4]);
ur=v(:,2)/fn/D;
uu=0.5*1000*D*L*v(:,2).^2;
CL_mean=(Lift_time_mean(2:end)-ll)./uu;
CL_fluctuate=Lift_fluctuate(2:end)./uu;
CD_mean=-(Drag_time_mean(2:end)-Drag_time_mean(1))./uu;
CD_fluctuate=Drag_fluctuate(2:end)./uu;
w=[ur, g(2:end), CL_mean, CL_fluctuate, CD_mean, CD_fluctuate, PhaseLag(2:end)];

scatter(w(:,1),w(:,2))
figure
subplot(2,2,1)
plot(w(:,1),w(:,3))
legend('CL mean')
subplot(2,2,2)
plot(w(:,1),w(:,4))
legend('CL fluctuate')
subplot(2,2,3)
plot(w(:,1),w(:,5))
legend('CD mean')
subplot(2,2,4)
plot(w(:,1),w(:,6))
legend('CD fluctuate')

figure
plot(w(:,1),w(:,7))


%% power spectral density

clear;clc;
% folder='E:\Files\experiment data\small k_0.8-8-120\VS frequency and displacement\T-0\L-4_210208\vsf_A';
% cd (folder)
foldername=dir('*.txt');
n=size(foldername,1); 
D=0.02;
L=0.325;
fn=1.88; % natural frequency in water
%  A
d=[];
for i=2:2:n
    m=foldername(i).name;
    a=importdata(m);
    b=(a.data(:,2)-mean(a.data(:,2)))*0.005/D;
    s1=length(b);
    c=sqrt(sum(b.^2)/s1)*sqrt(2);
    d=[d;c];
end
f=reshape(d,3,[]);
g=transpose(mean(f,1));

%vsf
fs=2000;
p1=[];
p2=[];
p3=[];
nfft=2^14;

for j1=1:6:n
    h1=foldername(j1).name;
    k1=importdata(h1);
    
    [pxx1,f1]=pwelch(k1.data(:,2),rectwin(nfft),0,nfft,fs);
    pxx1=pxx1/sum(pxx1(2:end)*mean(diff(f1)));
    f1=f1/fn;
    q1=[f1,pxx1];
    subplot(4,6,(j1+5)/6);loglog(q1(:,1),q1(:,2),'k')
    xlim([0 100]);
    p1=[p1,q1];
end

figure
j2=j1+2;
for j2=3:6:n
    h2=foldername(j2).name;
    k2=importdata(h2);
    
    [pxx2,f2]=pwelch(k2.data(:,2),rectwin(nfft),0,nfft,fs);
    pxx2=pxx2/sum(pxx2(2:end)*mean(diff(f2)));
    f2=f2/fn;
    q2=[f2,pxx2];
    subplot(4,6,(j2+3)/6);loglog(q2(:,1),q2(:,2),'k')
    xlim([0 100]);
    p2=[p2,q2];
end

figure
j3=j1+4;
for j3=5:6:n
    h3=foldername(j3).name;
    k3=importdata(h3);
    
    [pxx3,f3]=pwelch(k3.data(:,2),rectwin(nfft),0,nfft,fs);
    pxx3=pxx3/sum(pxx3(2:end)*mean(diff(f3)));
    f3=f3/fn;
    q3=[f3,pxx3];
    subplot(4,6,(j3+1)/6);loglog(q3(:,1),q3(:,2),'k')
    xlim([0 100]);
    p3=[p3,q3];
end

xlsxname=dir('*.xlsx');
t=importdata(xlsxname.name);
v=t.data.Sheet1(2:end,[1 4]);

ur=v(:,2)/fn/D;
w=[ur,g];

figure
scatter(w(:,1),w(:,2))
% %% 
idx=find(p1(:,1)>=100,1,'first');
pp=[p1(1:idx,[1 2:2:end]), p2(1:idx,2:2:end),p3(1:idx,2:2:end)];
% 
idy=find(foldername(2).name=='_');
savefile=strcat(foldername(2).name(1:idy(2)),'vsf.dat');
% 
save(savefile,'pp','-ascii')

%%
clear;clc;
% folder='D:\Documents\Nut\我的坚果云\results\results_fn_water_1.88\results_T-0\vsf';
% cd (folder)
foldername=dir('*.dat');
z=size(foldername,1); 

j=2;
m=foldername(j).name;
n=importdata(m);
c=(size(n,2)-1)/3;

for i1=2:c+1
    subplot(4,6,i1-1);loglog(n(:,1),n(:,i1),'k')
    xlim([0 100]);    
end
figure
for i2=c+2:2*c+1
    subplot(4,6,i2-c-1);loglog(n(:,1),n(:,i2),'k')
    xlim([0 100]);    
end
figure
for i3=2*c+2:3*c+1
    subplot(4,6,i3-2*c-1);loglog(n(:,1),n(:,i3),'k')
    xlim([0 100]);    
end

