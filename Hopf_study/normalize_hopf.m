function [w] =normalize_hopf(w)
  d=sum(abs(w),2);
%  keyboard
  d1=repmat(d,1,size(w,2));
    w=w./(d1+d1');

end