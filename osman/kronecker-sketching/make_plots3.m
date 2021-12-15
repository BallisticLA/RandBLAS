% Scripts used for making plots of a four and a nine with corresponding
% approximations.

n = 3;
width       = 1100;
height      = 180;

data_4_approx = double(tensor(cp_4));
data_9_approx = double(tensor(cp_9));

f1 = 1-data_4(3:end-2, 3:end-2, n);
f2 = 1-data_4_approx(3:end-2, 3:end-2, n);

f3 = 1-data_9(3:end-2, 3:end-2, n);
f4 = 1-data_9_approx(3:end-2, 3:end-2, n);

figure
subplot(1,4,1)
imshow(f1)
title('Exact 4')

subplot(1,4,2)
imshow(f2)
title('Approximate 4')

subplot(1,4,3)
imshow(f3)
title('Exact 9')

subplot(1,4,4)
imshow(f4)
title('Approximate 9')

colormap 'gray'
set(gcf,'position',[100,750,width,height])