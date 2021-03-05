function [f,power]=power_density(sig,fs)
N=length(sig);
sigfft=fft(sig)/N;
df=fs/N;
f=(0:df:df*N/2)';
power=(2*abs(sigfft(1:N/2+1)).^2)';
