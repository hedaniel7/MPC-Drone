function Draw_MPC_Overtaking(t,xx,xx1,u_cl,xs,N,w_lane,l_vehicle,w_vehicle)

set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 12)

line_width=1.5;
fontsize_labels=12;


%--------------------------------------------------------------------------
%-----------------------Simulate robots -----------------------------------
%--------------------------------------------------------------------------
x_r_1 = [];
y_r_1 = [];


figure(1)
% Animate the robot motion

set(gcf,'PaperPositionMode','auto')
set(gcf, 'Color', 'w');
set(gcf,'Units','normalized','OuterPosition',[0.25 0.5 0.55 0.4]);



for k = 1:size(xx,2)
%     plot([0 200],[0.5*w_lane 0.5*w_lane],'-g','linewidth',1.2);hold on % plot the reference trajectory
    plot([0 1000],[0.5*w_lane 0.5*w_lane],'-g','linewidth',1.2);hold on % plot the reference trajectory
    x1 = xx(1,k,1); 
    y1 = xx(2,k,1); 
    x_r_1 = [x_r_1 x1];
    y_r_1 = [y_r_1 y1];
    plot(x_r_1,y_r_1,'-r','linewidth',line_width);hold on % plot exhibited trajectory
    if k < size(xx,2) % plot prediction
        plot(xx1(1:N,1,k),xx1(1:N,2,k),'r--*')
    end
    
    %---  plot the boundary lane  ---
plot([-20 700],[0 0],'-k','linewidth',2);
hold on
plot([-20 700],[-0.1 -0.1],'-k','linewidth',2);
hold on
plot([-20 700],[w_lane w_lane],'--k','linewidth',2);
hold on
plot([-20 700],[2*w_lane 2*w_lane],'-k','linewidth',2);
hold on
plot([-20 700],[2*w_lane+0.1 2*w_lane+0.1],'-k','linewidth',2);
hold on
%     plot(x1,y1,'-sk','MarkerSize',20)% plot robot position
    rectangle('Position', [x1-l_vehicle/2 y1-w_vehicle/2 l_vehicle w_vehicle], 'EdgeColor', 'r', 'LineWidth', 1.25);
    hold off
    %figure(500)
    title(sprintf('$N=%d$',N),'interpreter','latex','FontSize',fontsize_labels)
    ylabel('$y$ (m)','interpreter','latex','FontSize',fontsize_labels)
%     ylabel('${a}_{y}$ (m/s^{2})','interpreter','latex','FontSize',fontsize_labels)
    xlabel('$x$ (m)','interpreter','latex','FontSize',fontsize_labels)
    axis([-1 700 -2 2*w_lane+2])
%     axis([x1-15 x1+30 -2 2*w_lane+2])
    pause(0.2)
%     box on;
%     grid on
    %aviobj = addframe(aviobj,gcf);
    drawnow
    % for video generation
    F(k) = getframe(gcf); % to get the current frame
end
saveas(gcf,'N30','epsc')
saveas(gcf,'N30','fig')
% close(gcf)
%viobj = close(aviobj)
%video = VideoWriter('exp.avi','Uncompressed AVI');

% video = VideoWriter('exp.avi','Motion JPEG AVI');
% video.FrameRate = 5;  % (frames per second) this number depends on the sampling time and the number of frames you have
% open(video)
% writeVideo(video,F)
% close (video)



% % fontsize_labels=12;
% figure(2)
% set(gcf,'Units','normalized','OuterPosition',[0.25 0.5 0.35 0.5]);
% subplot(211)
% stairs(t,u_cl(:,1),'k','linewidth',1.5); axis([0 t(end) -10 10])
% title(sprintf('$N=%d$',N),'interpreter','latex','FontSize',fontsize_labels)
% 
% ylabel('$a_{x} (m/s^{2})$','interpreter','latex','FontSize',fontsize_labels)
% grid on
% subplot(212)
% stairs(t,u_cl(:,2),'r','linewidth',1.5); axis([0 t(end) -0.85 0.85])
% xlabel('time $(s)$','interpreter','latex','FontSize',fontsize_labels)
% ylabel('$a_{y} (m/s^{2})$','interpreter','latex','FontSize',fontsize_labels)
% grid on
% % saveas(gcf,'N30u','epsc')
% % saveas(gcf,'N30u','fig')