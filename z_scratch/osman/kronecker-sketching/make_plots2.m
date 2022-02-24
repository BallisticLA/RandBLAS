% Make plots from mat files saved from experiment2.m

x           = 100:100:5e+3;
colors      = get(gca, 'colororder');
markers     = {'o', 'x', 's', 'd'};
line_styles = {'-', '--', ':', '-.'};
width       = 1100;
height      = 250;
no_sketches = 4;
close all

% -------------------------------------------------------------------------
% Figure 1: 4 and 9 from MNIST
% -------------------------------------------------------------------------

load('results_experiment2')
figure

error_dist  = abs(dist-true_dist)/true_dist;
mean_dist   = mean(error_dist,3);
std_dist    = std(error_dist,[],3);
max_dist    = max(error_dist,[],3);

subplot(1,3,1)
hold on
for k = 1:no_sketches
    plot(x, mean_dist(k,:), 'linewidth', 2, 'color', colors(k+1,:), 'linestyle', line_styles{k})
end
legend('KFJLT', 'TRP', 'TensorSketch', 'Sampling')
title('Mean')
xlabel('Embedding dimension')
axis tight

subplot(1,3,2)
hold on
for k = 1:no_sketches
    plot(x, std_dist(k,:), 'linewidth', 2, 'color', colors(k+1,:), 'linestyle', line_styles{k})
end
title('Standard deviation')
xlabel('Embedding dimension')
axis tight

subplot(1,3,3)
hold on
for k = 1:no_sketches
    plot(x, max_dist(k,:), 'linewidth', 2, 'color', colors(k+1,:), 'linestyle', line_styles{k})
end
title('Maximum')
xlabel('Embedding dimension')
axis tight

set(gcf,'position',[100,750,width,height])
