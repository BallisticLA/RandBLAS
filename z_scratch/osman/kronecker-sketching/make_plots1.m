% Make plots from mat files saved from experiment1.m

x       = 100:100:1000;
colors  = get(gca, 'colororder');
markers = {'+', 'o', 'x', 's', 'd'};
width   = 1100;
height  = 250;
close all

% -------------------------------------------------------------------------
% Figure 1: Normal
% -------------------------------------------------------------------------

load('results_experiment1_normal')
figure

subplot(1,3,1)
hold on
for k = 1:5
    plot(x, mean_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
legend('Gaussian', 'KFJLT', 'TRP', 'TensorSketch', 'Sampling')
title('Mean')
xlabel('Embedding dimension')
axis tight

subplot(1,3,2)
hold on
for k = 1:5
    plot(x, std_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
title('Standard deviation')
xlabel('Embedding dimension')
axis tight

subplot(1,3,3)
hold on
for k = 1:5
    plot(x, max_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
title('Maximum')
xlabel('Embedding dimension')
axis tight

set(gcf,'position',[100,750,width,height])


% -------------------------------------------------------------------------
% Figure 2: Sparse
% -------------------------------------------------------------------------

load('results_experiment1_sparse')
figure

subplot(1,3,1)
hold on
for k = 1:5
    plot(x, mean_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
legend('Gaussian', 'KFJLT', 'TRP', 'TensorSketch', 'Sampling')
title('Mean')
xlabel('Embedding dimension')
axis tight

subplot(1,3,2)
hold on
for k = 1:5
    plot(x, std_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
title('Standard deviation')
xlabel('Embedding dimension')
axis tight

subplot(1,3,3)
hold on
for k = 1:5
    plot(x, max_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
title('Maximum')
xlabel('Embedding dimension')
axis tight

set(gcf,'position',[100,400,width,height])


% -------------------------------------------------------------------------
% Figure 3: Single Element
% -------------------------------------------------------------------------

load('results_experiment1_large-single')
figure

subplot(1,3,1)
hold on
for k = 1:5
    plot(x, mean_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
legend('Gaussian', 'KFJLT', 'TRP', 'TensorSketch', 'Sampling')
title('Mean')
xlabel('Embedding dimension')
axis tight

subplot(1,3,2)
hold on
for k = 1:5
    plot(x, std_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
title('Standard deviation')
xlabel('Embedding dimension')
axis tight

subplot(1,3,3)
hold on
for k = 1:5
    plot(x, max_dist(k,:), 'linewidth', 2, 'marker', markers{k}, 'markersize', 6, 'color', colors(k,:))
end
title('Maximum')
xlabel('Embedding dimension')
axis tight

set(gcf,'position',[100,50,width,height])