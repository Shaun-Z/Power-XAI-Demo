
function mpc = Node8_loop
%CASE9    Power flow data for 8 bus distribution network
%   Please see CASEFORMAT for details on the case file format.
%

%   LYZ @ Apr 9th, 2015

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 1; %%% xxx


%% bus data
%   bus_i   type    Pd  Qd  Gs  Bs  area    Vm  Va  baseKV  zone    Vmax    Vmin
mpc.bus = [

	1	3	0.0	0.0	0	0	1	1.04	0.0	11	1	1.1	0.9;
	2	1	0.7	0.1	0	0	1	1.025	9.3	11	1	1.1	0.9;
	3	1	0.85	0.25	0	0	1	1.025	4.7	11	1	1.1	0.9;
	4	1	0.6	0.15	0	0	1	1.026	-2.2	11	1	1.1	0.9;
	5	1	1.25	0.5	0	0	1	0.996	-4.0	11	1	1.1	0.9;
	6	1	0.9	0.3	0	0	1	1.013	-3.7	11	1	1.1	0.9;
	7	1	0.1	0.1	0	0	1	1.026	3.7	11	1	1.1	0.9;
	8	1	1.0	0.35	0	0	1	1.016	0.7	11	1	1.1	0.9;
];

%   bus Pg  Qg  Qmax    Qmin    Vg  mBase   status  Pmax    Pmin    Pc1 Pc2 Qc1min  Qc1max  Qc2min  Qc2max  ramp_agc    ramp_10 ramp_30 ramp_q  apf
mpc.gen = [

	1	5.5	2.0	300	-300	1.04	1	1	250	0	0	0	0	0	0	0	0	0	0	0	0;
];

%% branch data
%   fbus    tbus    r   x   b   rateA   rateB   rateC   tap   angle   status  angmin  angmax
mpc.branch = [

	1	2	0.001008	0.005184	0.0	250	250	250	0	0	1	-360	360;
	2	3	0.001008	0.00765	0.01584	250	250	250	0	0	1	-360	360;
	3	4	0.00288	0.01449	0.02754	250	250	250	0	0	1	-360	360;
	4	5	0.00153	0.00828	0.01422	250	250	250	0	0	1	-360	360;
	5	6	0.00351	0.0153	0.03222	250	250	250	0	0	1	-360	360;
	4	7	0.00153	0.00648	0.01341	250	250	250	0	0	1	-360	360;
	7	8	0.001071	0.009072	0.01881	250	250	250	0	0	1	-360	360;
    2   6   0.002008	0.007184	0.0	250	250	250	0	0	1	-360	360;
    1	8	0.001008	0.007184	0.01881	250	250	250	0	0	1	-360	360;
%    3	5	0.001008	0.0153	0.01341	250	250	250	0	0	1	-360	360;
];
