clear; clc; close all;
rng(1);

Ts = 0.01;
T  = 20;
N  = round(T/Ts);
t  = (0:N-1)*Ts;

Xu  = -1.982e3;  Xw  =  4.025e3;
Zu  = -2.595e4;  Zw  = -9.030e4;  Zq = -4.524e5;  Zwd = 1.909e3;
Mu  =  1.593e4;  Mw  = -1.563e5;  Mq = -1.521e7;  Mwd = -1.702e4;
g    = 9.81;     theta0 = 0;
S    = 511;      cbar = 8.324;    U0 = 235.9;
Iyy  = 0.449e8;  m    = 2.83176e6/g;   rho = 0.3045;

Xdp = 0.3*m*g;   Zdp = 0;   Mdp = 0;
Xde = -3.818e-6 * (0.5*rho*U0^2*S);
Zde = -0.3648   * (0.5*rho*U0^2*S);
Mde = -1.444    * (0.5*rho*U0^2*S*cbar);

A = [ Xu/m,                      Xw/m,                          0,                                 -g;
      Zu/(m-Zwd),                Zw/(m-Zwd),                    (Zq+m*U0)/(m-Zwd),                  0;
      (Mu+Zu*Mwd/(m-Zwd))/Iyy,  (Mw+Zw*Mwd/(m-Zwd))/Iyy,      (Mq+(Zq+m*U0)*Mwd/(m-Zwd))/Iyy,   0;
      0,                         0,                             1,                                  0 ];

A_aug = [ A,          zeros(4,1);
          0, -1, 0,  U0, 0 ];

B_aug = [ Xde/m,                        Xdp/m;
          Zde/(m-Zwd),                  Zdp/(m-Zwd);
          (Mde+Zde*Mwd/(m-Zwd))/Iyy,   (Mdp+Zdp*Mwd/(m-Zwd))/Iyy;
          0,                            0;
          0,                            0 ];

Bw_aug = [ -Xu/m,                       -Xw/m,                        0;
           -Zu/(m-Zwd),                 -Zw/(m-Zwd),                  0;
           (-Mu-Zu*Mwd/(m-Zwd))/Iyy,   (-Mw-Zw*Mwd/(m-Zwd))/Iyy,   -Mq/Iyy;
            0,                           0,                            0;
            0,                           0,                            0 ];

sysd = c2d(ss(A_aug, [B_aug, Bw_aug], eye(5), 0), Ts);
Ad   = sysd.A;
Bd   = sysd.B(:, 1:2);
Bwd  = sysd.B(:, 3:5);

C = [ 1 0 0 0 0;
      0 1 0 0 0;
      0 0 1 0 0;
      0 0 0 1 0 ];

x_true = zeros(5, N);
x_kf   = zeros(5, N);
x_gru2 = zeros(5, N);
x_gru1 = zeros(5, N);

x0 = [5; 10; 0.01; 0.08; 200];
x_true(:,1) = x0;
x_kf(:,1)   = x0;
x_gru2(:,1) = x0;
x_gru1(:,1) = x0;

u    = zeros(2, N);
udot = zeros(2, N);
for k = 2:N
    udot(:,k) = 0.01 * randn(2,1);
    u(:,k)    = u(:,k-1) + Ts*udot(:,k);
end

gust = @(tk) [ 10*sin(2*pi*tk) * (tk>=10 && tk<=15);
                5*cos(2*pi*tk) * (tk>=10 && tk<=15);
                1.5             * (tk>=10 && tk<=15) ];

Q  = diag([1e-7, 1e-7, 1e-9, 1e-10, 1e-10]);
R0 = diag([(1e-1)^2, (1e-1)^2, (1e-1)^2, 1]);
R  = R0;
P  = eye(5);

win        = 50;
lambda_f   = 0.95;
R_floor    = 1e-8;
innov_buf  = zeros(4, win);
R_hist     = zeros(4, N);
innov_hist = zeros(4, N);
R_hist(:,1) = diag(R);

for k = 2:N
    tk = t(k-1);
    wg = gust(tk);

    x_true(:,k) = Ad*x_true(:,k-1) + Bd*u(:,k-1) + Bwd*wg;

    y = C*x_true(:,k) + 0.02*randn(4,1);

    x_pred = Ad*x_kf(:,k-1) + Bd*u(:,k-1);
    P_pred = Ad*P*Ad' + Q;

    S_pred_full = C*P_pred*C' + R;
    K_gain      = P_pred*C' / S_pred_full;
    innov       = y - C*x_pred;
    x_kf(:,k)   = x_pred + K_gain*innov;
    P           = (eye(5) - K_gain*C) * P_pred;

    innov_hist(:,k) = innov;

    innov_buf = [innov_buf(:,2:end), innov];
    if k > win
        R_emp_diag  = var(innov_buf, 0, 2);
        S_pred_diag = diag(S_pred_full);
        for j = 1:4
            if R_emp_diag(j) > S_pred_diag(j)
                R(j,j) = lambda_f*R(j,j) + (1-lambda_f)*R_emp_diag(j);
            else
                R(j,j) = lambda_f*R(j,j) + (1-lambda_f)*R0(j,j);
            end
            R(j,j) = max(R(j,j), R_floor);
        end
    end
    R_hist(:,k) = diag(R);
end

sig      = @(x) 1./(1+exp(-x));
sig_der  = @(f) f.*(1-f);
tanh_der = @(f) 1 - f.^2;

nx = 7;
ny = 5;

all_inputs = zeros(nx, N-2);
for kk = 2:N-1
    all_inputs(:, kk-1) = [x_kf(:,kk); udot(:,kk)];
end
xmin = min(all_inputs, [], 2);
xmax = max(all_inputs, [], 2);
xrng = xmax - xmin;
xrng(xrng < 1e-8) = 1e-8;

target_scale = [5; 5; 0.1; 0.1; 500];
epochs = 100;
lr     = 1e-4;

nh1 = 8; nh2 = 16;

lim1 = sqrt(6/(nh1+nx));
Wr1=-lim1+2*lim1*rand(nh1,nx); Ur1=-lim1+2*lim1*rand(nh1,nh1); br1=zeros(nh1,1);
Wz1=-lim1+2*lim1*rand(nh1,nx); Uz1=-lim1+2*lim1*rand(nh1,nh1); bz1=zeros(nh1,1);
Wh1=-lim1+2*lim1*rand(nh1,nx); Uh1=-lim1+2*lim1*rand(nh1,nh1); bh1=zeros(nh1,1);

lim2 = sqrt(6/(nh2+nh1));
Wr2=-lim2+2*lim2*rand(nh2,nh1); Ur2=-lim2+2*lim2*rand(nh2,nh2); br2=zeros(nh2,1);
Wz2=-lim2+2*lim2*rand(nh2,nh1); Uz2=-lim2+2*lim2*rand(nh2,nh2); bz2=zeros(nh2,1);
Wh2=-lim2+2*lim2*rand(nh2,nh1); Uh2=-lim2+2*lim2*rand(nh2,nh2); bh2=zeros(nh2,1);

Wy2 = 0.01*randn(ny, nh2);
by2 = zeros(ny, 1);

loss_hist_2 = zeros(epochs,1);

for ep = 1:epochs
    h1 = zeros(nh1,1); h2 = zeros(nh2,1);
    total_loss = 0;

    for k = 2:N-1
        inp      = [x_kf(:,k); udot(:,k)];
        inp_norm = 2*(inp - xmin)./xrng - 1;

        h1_prev = h1; h2_prev = h2;

        r1 = sig( Wr1*inp_norm + Ur1*h1_prev + br1 );
        z1 = sig( Wz1*inp_norm + Uz1*h1_prev + bz1 );
        h_tilde1 = tanh( Wh1*inp_norm + Uh1*(r1.*h1_prev) + bh1 );
        h1_new = (1-z1).*h1_prev + z1.*h_tilde1;

        r2 = sig( Wr2*h1_new + Ur2*h2_prev + br2 );
        z2 = sig( Wz2*h1_new + Uz2*h2_prev + bz2 );
        h_tilde2 = tanh( Wh2*h1_new + Uh2*(r2.*h2_prev) + bh2 );
        h2_new = (1-z2).*h2_prev + z2.*h_tilde2;

        y_hat  = Wy2*h2_new + by2;
        y_true = (x_true(:,k+1) - (Ad*x_kf(:,k) + Bd*u(:,k))) ./ target_scale;

        e = y_hat - y_true;
        total_loss = total_loss + e'*e;

        dWy2 = e * h2_new';  dby2 = e;
        dh2  = Wy2' * e;

        dz2       = dh2 .* (h_tilde2 - h2_prev) .* sig_der(z2);
        dh_tilde2 = dh2 .* z2 .* tanh_der(h_tilde2);
        dr2       = (Uh2' * dh_tilde2) .* h2_prev .* sig_der(r2);

        Wy2 = Wy2 - lr*dWy2;  by2 = by2 - lr*dby2;
        Wz2 = Wz2 - lr*(dz2*h1_new');       Uz2 = Uz2 - lr*(dz2*h2_prev');            bz2 = bz2 - lr*dz2;
        Wh2 = Wh2 - lr*(dh_tilde2*h1_new'); Uh2 = Uh2 - lr*(dh_tilde2*(r2.*h2_prev)'); bh2 = bh2 - lr*dh_tilde2;
        Wr2 = Wr2 - lr*(dr2*h1_new');       Ur2 = Ur2 - lr*(dr2*h2_prev');            br2 = br2 - lr*dr2;

        dh1 = Wz2'*dz2 + Wh2'*dh_tilde2 + Wr2'*dr2;

        dz1       = dh1 .* (h_tilde1 - h1_prev) .* sig_der(z1);
        dh_tilde1 = dh1 .* z1 .* tanh_der(h_tilde1);
        dr1       = (Uh1' * dh_tilde1) .* h1_prev .* sig_der(r1);

        Wz1 = Wz1 - lr*(dz1*inp_norm');       Uz1 = Uz1 - lr*(dz1*h1_prev');            bz1 = bz1 - lr*dz1;
        Wh1 = Wh1 - lr*(dh_tilde1*inp_norm'); Uh1 = Uh1 - lr*(dh_tilde1*(r1.*h1_prev)'); bh1 = bh1 - lr*dh_tilde1;
        Wr1 = Wr1 - lr*(dr1*inp_norm');       Ur1 = Ur1 - lr*(dr1*h1_prev');            br1 = br1 - lr*dr1;

        h1 = h1_new; h2 = h2_new;
    end

    loss_hist_2(ep) = total_loss/(N-2);
    fprintf('[2-layer GRU] Epoch %3d | Loss = %.6f\n', ep, loss_hist_2(ep));
end

h1 = zeros(nh1,1); h2 = zeros(nh2,1);
for k = 2:N-1
    inp      = [x_kf(:,k); udot(:,k)];
    inp_norm = 2*(inp - xmin)./xrng - 1;

    r1 = sig( Wr1*inp_norm + Ur1*h1 + br1 );
    z1 = sig( Wz1*inp_norm + Uz1*h1 + bz1 );
    h_tilde1 = tanh( Wh1*inp_norm + Uh1*(r1.*h1) + bh1 );
    h1 = (1-z1).*h1 + z1.*h_tilde1;

    r2 = sig( Wr2*h1 + Ur2*h2 + br2 );
    z2 = sig( Wz2*h1 + Uz2*h2 + bz2 );
    h_tilde2 = tanh( Wh2*h1 + Uh2*(r2.*h2) + bh2 );
    h2 = (1-z2).*h2 + z2.*h_tilde2;

    y_hat = Wy2*h2 + by2;
    dx    = y_hat .* target_scale;
    x_gru2(:,k+1) = Ad*x_kf(:,k) + Bd*u(:,k) + dx;
end
x_gru2(:,N) = x_gru2(:,N-1);

nh = 8;

lim = sqrt(6/(nh+nx));
Wr=-lim+2*lim*rand(nh,nx); Ur=-lim+2*lim*rand(nh,nh); br=zeros(nh,1);
Wz=-lim+2*lim*rand(nh,nx); Uz=-lim+2*lim*rand(nh,nh); bz=zeros(nh,1);
Wh=-lim+2*lim*rand(nh,nx); Uh=-lim+2*lim*rand(nh,nh); bh=zeros(nh,1);

Wy1 = 0.01*randn(ny, nh);
by1 = zeros(ny, 1);

loss_hist_1 = zeros(epochs,1);

for ep = 1:epochs
    h = zeros(nh,1);
    total_loss = 0;

    for k = 2:N-1
        inp      = [x_kf(:,k); udot(:,k)];
        inp_norm = 2*(inp - xmin)./xrng - 1;

        h_prev = h;
        r = sig( Wr*inp_norm + Ur*h_prev + br );
        z = sig( Wz*inp_norm + Uz*h_prev + bz );
        h_tilde = tanh( Wh*inp_norm + Uh*(r.*h_prev) + bh );
        h_new = (1-z).*h_prev + z.*h_tilde;

        y_hat  = Wy1*h_new + by1;
        y_true = (x_true(:,k+1) - (Ad*x_kf(:,k) + Bd*u(:,k))) ./ target_scale;

        e = y_hat - y_true;
        total_loss = total_loss + e'*e;

        dWy1 = e * h_new';  dby1 = e;
        dh   = Wy1' * e;

        dz       = dh .* (h_tilde - h_prev) .* sig_der(z);
        dh_tilde = dh .* z .* tanh_der(h_tilde);
        dr       = (Uh' * dh_tilde) .* h_prev .* sig_der(r);

        Wy1 = Wy1 - lr*dWy1;  by1 = by1 - lr*dby1;
        Wz  = Wz  - lr*(dz*inp_norm');       Uz = Uz - lr*(dz*h_prev');            bz = bz - lr*dz;
        Wh  = Wh  - lr*(dh_tilde*inp_norm'); Uh = Uh - lr*(dh_tilde*(r.*h_prev)'); bh = bh - lr*dh_tilde;
        Wr  = Wr  - lr*(dr*inp_norm');       Ur = Ur - lr*(dr*h_prev');            br = br - lr*dr;

        h = h_new;
    end

    loss_hist_1(ep) = total_loss/(N-2);
    fprintf('[1-layer GRU] Epoch %3d | Loss = %.6f\n', ep, loss_hist_1(ep));
end

h = zeros(nh,1);
for k = 2:N-1
    inp      = [x_kf(:,k); udot(:,k)];
    inp_norm = 2*(inp - xmin)./xrng - 1;

    r = sig( Wr*inp_norm + Ur*h + br );
    z = sig( Wz*inp_norm + Uz*h + bz );
    h_tilde = tanh( Wh*inp_norm + Uh*(r.*h) + bh );
    h = (1-z).*h + z.*h_tilde;

    y_hat = Wy1*h + by1;
    dx    = y_hat .* target_scale;
    x_gru1(:,k+1) = Ad*x_kf(:,k) + Bd*u(:,k) + dx;
end
x_gru1(:,N) = x_gru1(:,N-1);

state_labels = {'u','w','q','theta','h'};
idx = 2:N;

rmse_kf = zeros(5,1); rmse_g1 = zeros(5,1); rmse_g2 = zeros(5,1);
for j = 1:5
    rmse_kf(j) = sqrt(mean((x_true(j,idx) - x_kf(j,idx)).^2));
    rmse_g1(j) = sqrt(mean((x_true(j,idx) - x_gru1(j,idx)).^2));
    rmse_g2(j) = sqrt(mean((x_true(j,idx) - x_gru2(j,idx)).^2));
end
pct_improve_2 = 100*(rmse_kf - rmse_g2)./rmse_kf;

fprintf('\n================= Table 1: RMSE Comparison =================\n');
fprintf('%6s | %10s | %10s | %10s | %12s\n', 'State','KF','KF+1L-GRU','KF+2L-GRU','%%improve(2L)');
for j = 1:5
    fprintf('%6s | %10.4f | %10.4f | %10.4f | %11.1f%%\n', ...
        state_labels{j}, rmse_kf(j), rmse_g1(j), rmse_g2(j), pct_improve_2(j));
end

nparams_1 = 3*(nh*nx + nh*nh + nh) + (ny*nh + ny);
nparams_2 = 3*(nh1*nx + nh1*nh1 + nh1) + 3*(nh2*nh1 + nh2*nh2 + nh2) + (ny*nh2 + ny);
fprintf('\nTrainable parameters -- 1-layer GRU: %d | 2-layer GRU: %d\n', nparams_1, nparams_2);

labels = {'u (m/s)', 'w (m/s)', 'q (rad/s)', '\theta (rad)', 'h (m)'};

outdir = 'figures';
if ~exist(outdir, 'dir'); mkdir(outdir); end

lw       = 2.0;
col_true = [0    0    0   ];
col_kf   = [0.85 0.325 0.098];
col_g1   = [0.466 0.674 0.188];
col_g2   = [0    0.447 0.741];
col_R    = [0.494 0.184 0.556];
col_i1   = [0    0.447 0.741];
col_i2   = [0.85 0.325 0.098];
col_gust = [0.85 0.85 0.85];

figure('Name','fig1_state_estimation','NumberTitle','off');
for i = 1:5
    subplot(5,1,i)
    plot(t, x_true(i,:), 'Color', col_true, 'LineWidth', lw); hold on;
    plot(t, x_kf(i,:),   'Color', col_kf,   'LineWidth', lw);
    plot(t, x_gru2(i,:), 'Color', col_g2,   'LineWidth', lw);
    ylabel(labels{i}); box on;
    if i == 1
        legend('True','KF','KF+2L-GRU','Location','best','Box','off');
    end
end
xlabel('Time (s)');
saveFig(gcf, 'fig1_state_estimation', outdir);

figure('Name','fig2_depth_ablation','NumberTitle','off','Position',[100 100 800 700]);
for i = 1:5
    subplot(5,1,i)
    plot(t, x_true(i,:), 'Color', col_true, 'LineWidth', lw); hold on;
    plot(t, x_kf(i,:),   'Color', col_kf,   'LineWidth', lw);
    plot(t, x_gru1(i,:), 'Color', col_g1,   'LineWidth', lw);
    plot(t, x_gru2(i,:), 'Color', col_g2,   'LineWidth', lw);
    ylabel(labels{i}); box on;
    if i == 1
        legend('True','KF','KF+1L-GRU','KF+2L-GRU','Location','best','Box','off');
    end
end
xlabel('Time (s)');
saveFig(gcf, 'fig2_depth_ablation', outdir);

err_kf   = abs(x_true - x_kf);
err_gru2 = abs(x_true - x_gru2);
figure('Name','fig3_absolute_error','NumberTitle','off','Position',[100 100 800 600]);
for i = 1:5
    subplot(5,1,i)
    plot(t, err_kf(i,:),   'Color', col_kf, 'LineWidth', lw); hold on;
    plot(t, err_gru2(i,:), 'Color', col_g2, 'LineWidth', lw);
    ylabel(['|', labels{i}, ' - est|']); box on;
    if i == 1
        legend('KF Error','KF+2L-GRU Error','Location','best','Box','off');
    end
    yl = ylim;
    patch([10 15 15 10],[yl(1) yl(1) yl(2) yl(2)], col_gust, 'EdgeColor','none','FaceAlpha',0.5);
    uistack(findobj(gca,'Type','patch'),'bottom');
end
xlabel('Time (s)');
saveFig(gcf, 'fig3_absolute_error', outdir);

zoom_idx = t >= 8 & t <= 17;
figure('Name','fig4_zoomed_gust_window','NumberTitle','off','Position',[150 150 700 500]);
subplot(2,1,1)
plot(t(zoom_idx), x_true(2,zoom_idx), 'Color', col_true, 'LineWidth', lw); hold on;
plot(t(zoom_idx), x_kf(2,zoom_idx),   'Color', col_kf,   'LineWidth', lw);
plot(t(zoom_idx), x_gru1(2,zoom_idx), 'Color', col_g1,   'LineWidth', lw);
plot(t(zoom_idx), x_gru2(2,zoom_idx), 'Color', col_g2,   'LineWidth', lw);
ylabel('w (m/s)'); box on;
legend('True','KF','KF+1L-GRU','KF+2L-GRU','Location','best','Box','off');

subplot(2,1,2)
plot(t(zoom_idx), x_true(3,zoom_idx), 'Color', col_true, 'LineWidth', lw); hold on;
plot(t(zoom_idx), x_kf(3,zoom_idx),   'Color', col_kf,   'LineWidth', lw);
plot(t(zoom_idx), x_gru1(3,zoom_idx), 'Color', col_g1,   'LineWidth', lw);
plot(t(zoom_idx), x_gru2(3,zoom_idx), 'Color', col_g2,   'LineWidth', lw);
ylabel('q (rad/s)'); xlabel('Time (s)'); box on;
saveFig(gcf, 'fig4_zoomed_gust_window', outdir);

figure('Name','fig5_training_convergence','NumberTitle','off');
semilogy(1:epochs, loss_hist_1, 'Color', col_g1, 'LineWidth', lw); hold on;
semilogy(1:epochs, loss_hist_2, 'Color', col_g2, 'LineWidth', lw);
xlabel('Epoch'); ylabel('Normalized MSE Loss'); box on;
legend('1-layer GRU','2-layer GRU','Location','best','Box','off');
saveFig(gcf, 'fig5_training_convergence', outdir);

meas_labels = {'u','w','q','\theta'};
figure('Name','fig6_adaptive_R','NumberTitle','off','Position',[150 150 700 600]);
for i = 1:4
    subplot(4,1,i)
    plot(t, R_hist(i,:), 'Color', col_R, 'LineWidth', lw); hold on;
    ylabel(['R_{' meas_labels{i} '}']); box on;
    yl = ylim;
    patch([10 15 15 10],[yl(1) yl(1) yl(2) yl(2)], col_gust, 'EdgeColor','none','FaceAlpha',0.5);
    uistack(findobj(gca,'Type','patch'),'bottom');
end
xlabel('Time (s)');
saveFig(gcf, 'fig6_adaptive_R', outdir);

figure('Name','fig7_innovation_whitening','NumberTitle','off','Position',[150 150 700 600]);
subplot(2,1,1)
plot(t, innov_hist(1,:), 'Color', col_i1, 'LineWidth', 1.2); hold on;
plot(t, innov_hist(3,:), 'Color', col_i2, 'LineWidth', 1.2);
ylabel('Innovation'); box on;
legend('u channel','q channel','Location','best','Box','off');

pre_gust_idx = t < 9;
seg = innov_hist(1, pre_gust_idx) - mean(innov_hist(1, pre_gust_idx));
maxlag = 100;
acf = zeros(2*maxlag+1, 1);
for lg = -maxlag:maxlag
    if lg >= 0
        acf(lg+maxlag+1) = sum(seg(1:end-lg) .* seg(1+lg:end));
    else
        acf(lg+maxlag+1) = sum(seg(1-lg:end) .* seg(1:end+lg));
    end
end
acf = acf / acf(maxlag+1);
lags = -maxlag:maxlag;

subplot(2,1,2)
stem(lags, acf, 'filled', 'MarkerSize', 2, 'Color', col_g2);
xlabel('Lag (samples)'); ylabel('Autocorrelation'); box on;
saveFig(gcf, 'fig7_innovation_whitening', outdir);

figure('Name','fig8_rmse_comparison','NumberTitle','off','Position',[200 150 750 400]);
bar_data = [rmse_kf, rmse_g1, rmse_g2];
hb = bar(bar_data);
hb(1).FaceColor = col_kf;
hb(2).FaceColor = col_g1;
hb(3).FaceColor = col_g2;
hold on;
xticks(1:5); xticklabels(state_labels);
ylabel('RMSE'); box on;
legend('KF','KF+1L-GRU','KF+2L-GRU','Location','northoutside','Orientation','horizontal','Box','off');
for i = 1:5
    for kcol = 1:3
        xloc = hb(kcol).XEndPoints(i);
        yloc = hb(kcol).YEndPoints(i);
        text(xloc, yloc + 0.02*max(bar_data(:)), sprintf('%.3g', bar_data(i,kcol)), ...
             'HorizontalAlignment','center','FontSize',8);
    end
end
saveFig(gcf, 'fig8_rmse_comparison', outdir);

fprintf('\nAll 8 figures saved to ./%s as .png (300 dpi) and .pdf (vector).\n', outdir);

function saveFig(figHandle, name, outdir)
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    if isempty(figHandle) || ~isgraphics(figHandle) || ~isvalid(figHandle)
        warning('saveFig: handle for "%s" is not a valid figure -- skipping save.', name);
        return;
    end

    set(figHandle, 'Color', 'w');
    axesList = findall(figHandle, 'Type', 'axes');
    set(axesList, 'Color', 'w', 'GridLineStyle', 'none', 'XGrid','off','YGrid','off');
    drawnow;

    pngFile = fullfile(outdir, [name '.png']);
    pdfFile = fullfile(outdir, [name '.pdf']);

    try
        exportgraphics(figHandle, pngFile, 'Resolution', 300);
    catch ME1
        warning('exportgraphics PNG failed for "%s" (%s) -- using print() fallback.', name, ME1.message);
        print(figHandle, pngFile, '-dpng', '-r300');
    end

    try
        exportgraphics(figHandle, pdfFile, 'ContentType', 'vector');
    catch ME2
        warning('exportgraphics PDF failed for "%s" (%s) -- using print() fallback.', name, ME2.message);
        print(figHandle, pdfFile, '-dpdf', '-bestfit');
    end
end