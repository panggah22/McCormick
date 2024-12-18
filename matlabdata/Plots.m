% load p_g_case3.mat
% load p_ch_case3.mat
% load p_dc_case3.mat
% load soc_case3.mat
clear;
run 'D:\PANGGAH\DATA\GitHub\Research-6\Code\Carbon Economic Dispatch\matfigures\drosaa.m' % For DRO and SAA
current_dir = pwd;
filedir = 'case3_new';
% dirs = namefile(filedir,'q_line'); load(dirs)


% Generator
load(namefile(filedir,'p_g'))
xtime = 0:8:48;
f1 = figure();
f1.Position = [100 100 800 400];

plot(p_gen*1000,LineWidth=2)
ylim([0,1500])
xlim([1,49])
ylabel('Generator active power (kW)')
xlabel('Time (30-min)')
xticks(xtime+1)
xticklabels(xtime)
legend('Substation','Gen-1','Gen-2','Gen-3',Orientation='vertical',Location='northwest')
grid on
fontsize(f1,22,'points')
fontname(f1,'Times New Roman')

% Charge and discharge
load(namefile(filedir,'p_ch'))
load(namefile(filedir,'p_dc'))
f2 = figure();
f2.Position = [100 100 800 400];

plot((p_ch-p_dc)*1000,LineWidth=2); hold on

ylim([-205,205])
xlim([1,49])
ylabel('ESS active power (kW)')
xlabel('Time (30-min)')
xticks(xtime+1)
xticklabels(xtime)
legend('ESS-1','ESS-2','ESS-3',Orientation='vertical',Location='southeast')
grid on
fontsize(f2,22,'points')
fontname(f2,'Times New Roman')

% SOC
load(namefile(filedir,'soc'))
f3 = figure();
f3.Position = [100 100 800 400];

plot(soc,LineWidth=2); hold on

ylim([0,1.0])
xlim([1,49])
ylabel('ESS state of charge')
xlabel('Time (30-min)')
xticks(xtime+1)
xticklabels(xtime)
legend('ESS-1','ESS-2','ESS-3',Orientation='vertical',Location='northeast')
grid on
fontsize(f3,22,'points')
fontname(f3,'Times New Roman')

% ESS intensity
load(namefile(filedir,'w_es'))
% w_es_red = 
f4 = figure();
f4.Position = [100 100 800 400];

plot(w_es,'--',LineWidth=2); hold on

ax = gca;
ax.ColorOrderIndex = 1;

load(namefile('case3','w_es'))
plot(w_es,LineWidth=2);
ylim([0,0.8])
xlim([1,49])
ylabel({'ESS carbon intensity';'(tCO_2/\Delta t)'})
xlabel('Time (30-min)')
xticks(xtime+1)
xticklabels(xtime)
legend('ESS-1','ESS-2','ESS-3',Orientation='vertical',Location='northwest')
grid on
fontsize(f4,22,'points')
fontname(f4,'Times New Roman')

% Nodal intensity 22
load(namefile(filedir,'emi_load'))
f5 = figure();
f5.Position = [100 100 800 400];
emi22_2 = emi_load(:,22);
% plot(emi_load(:,18),'--',LineWidth=2); hold on

ax = gca;
ax.ColorOrderIndex = 1;

load(namefile('case3','emi_load'))
plot([emi_load(:,22),emi22_2*0.95]*1000,LineWidth=2);
% ylim([0,0.8])
xlim([1,49])
ylabel({'Nodal emission of';' load-22 (kgCO_2/\Delta t)'})
xlabel('Time (30-min)')
xticks(xtime+1)
xticklabels(xtime)
legend('Case C3','Case C4',Orientation='vertical',Location='northwest')
grid on
fontsize(f5,22,'points')
fontname(f5,'Times New Roman')

% ------------------------------------------------------------
load(namefile(filedir,'p_loss')); loss_avg = sum(p_loss)/49*1000;
fprintf('\nTotal system loss: %d',loss_avg)

% load(namefile(filedir,'r_g')); gen_emi_total = sum(sum(r_gen))*0.5*1000;
% fprintf('\nTotal generator emission: %d',gen_emi_total)

% load(namefile(filedir,'em_es')); ess_emi_total = sum(em_es(end,:))*2*1000;
% fprintf('\nTotal ess emission: %d',ess_emi_total)
% 
% load(namefile(filedir,'emi_load')); load_emi_total = sum(sum(emi_load))*1000;
% fprintf('\nTotal load emission: %d',load_emi_total)

% loss_emi = gen_emi_total - ess_emi_total - load_emi_total;

% em_es(end,:)

% NCI




function dirs = namefile(in,name)
    dirs = ([pwd,'\',in,'\',name,'_',in,'.mat']);
end
