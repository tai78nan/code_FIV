function [amp,pha,f,d]=amp_pha(sig,fs)
sigfft=fft(sig);
df=fs/length(sig);
amp=abs(sigfft)*2/length(sig);
f=(0:df:df*length(sig)/2)';
pha=angle(sigfft);
% pha(find(amp<1))=0;
d = amp(1:length(sig)/2+1);
figure;loglog(f,d,'k');xlim([0 100]);

