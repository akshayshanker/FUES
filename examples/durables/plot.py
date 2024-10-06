import seaborn as sns
import matplotlib.pylab as pl
from matplotlib.ticker import FormatStrFormatter
import numpy as np

def plot_pols(cp, Results1, Results2, plot_t, index):

		pl.close()
		sns.set(
			style="white", rc={
				"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})
		
		palette = sns.color_palette("cubehelix", 3)
		palette1 = sns.color_palette("cubehelix", 5)
		color1 = palette[0] 
		color2 = palette[1]
		color3 = palette[2]
		palette[2] = palette1[0]
		colors = palette
		colors2 = palette1
		labs = ['H = Low', ' H = Med.', ' H = High']

		fig_pol, ax_pol = pl.subplots(1, 2, figsize=(8, 6))
		fig_pol_a, ax_pol_a = pl.subplots(1, 2, figsize=(8, 6))
		fig_val, ax_val = pl.subplots(1, 2, figsize=(8, 6))
		fig_pol_a, ax_pol_a = pl.subplots(1, 2, figsize=(8, 6))
		fig_pol_c, ax_pol_c = pl.subplots(1, 2, figsize=(8, 6))

		for i_z in [0,1]:
			for col_ih, i_h, lbs in zip([0, 1, 2], index, labs):

				pos_col = np.where(
					np.abs(np.diff(Results1[plot_t]["Hadj"][i_z, :])) > 1e100)[0] + 1
				g_1 = np.insert(
					Results1[plot_t]["Hadj"][i_z, :], pos_col, np.nan)
				
				x1 = np.insert(cp.asset_grid_WE, pos_col, np.nan)

				pos_bell = np.where(
					np.abs(np.diff(Results2[plot_t]["Hadj"][i_z, :])) > 1e100)[0] + 1
				g_2 = np.insert(
					Results2[plot_t]["Hadj"][i_z, :], pos_bell, np.nan)
				x2 = np.insert(cp.asset_grid_WE, pos_bell, np.nan)

				ax_pol[0].plot(x1, g_1, color=colors[col_ih],
							   label=lbs,
							   linewidth=0.75)
				ax_pol[1].plot(x2, g_2, color=colors[col_ih],
							   label=lbs,
							   linewidth=0.75)
				
				ax_val[1].plot(cp.asset_grid_A, Results1[plot_t]["VF"][i_z, :, i_h],
							   color=colors[col_ih])
				
				ax_val[1].plot(cp.asset_grid_A,
							    Results2[plot_t]["VF"]
										 [i_z,
										 :, i_h],
							   linestyle='--',
							   color=colors2[col_ih],
							   linewidth=0.75)

				ax_pol[0].spines['right'].set_visible(False)
				ax_pol[0].spines['top'].set_visible(False)
				ax_val[0].spines['right'].set_visible(False)
				ax_val[0].spines['top'].set_visible(False)
				ax_val[0].grid(True)
				ax_pol[0].set_yticklabels(ax_pol[0].get_yticks(), size=9)
				ax_pol[0].set_xticklabels(ax_pol[0].get_xticks(), size=9)
				ax_pol[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
				ax_pol[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
				ax_pol[1].set_xlabel(r'Start of time $t$ financial assets', fontsize=11)
				ax_pol[1].set_ylabel(r'End of time $t$ housing assets', fontsize=11)
				ax_pol[0].set_xlabel(r'Start of time $t$ financial assets', fontsize=11)
				ax_pol[0].set_ylabel(r'End of time $t$ housing assets', fontsize=11)
				ax_pol[1].spines['right'].set_visible(False)
				ax_pol[1].spines['top'].set_visible(False)
				ax_pol[0].spines['right'].set_visible(False)
				ax_pol[0].spines['top'].set_visible(False)
				ax_pol[1].set_yticklabels(ax_pol[0].get_yticks(), size=9)
				ax_pol[1].set_xticklabels(ax_pol[0].get_xticks(), size=9)
				ax_pol[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
				ax_pol[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
				ax_pol[1].grid(True)
				ax_pol[0].grid(True)

				pos_col_a = np.where(
					np.abs(np.diff(Results1[plot_t]["Aadj"][i_z, :])) > 0.08)[0] + 1
				g_1a = np.insert(
					Results1[plot_t]["Aadj"][i_z, :], pos_col_a, np.nan)
				x1a = np.insert(cp.asset_grid_WE, pos_col_a, np.nan)

				pos_bell_a = np.where(
					np.abs(np.diff(Results2[plot_t]["Aadj"][i_z, :])) > 2)[0] + 1
				g_2a = np.insert(
					Results2[plot_t]["Aadj"][i_z, :], pos_bell_a, np.nan)
				x2a = np.insert(cp.asset_grid_WE, pos_bell_a, np.nan)

				ax_pol_a[0].plot(x1a, g_1a, color=colors[col_ih],
								 label=lbs,
								 linewidth=0.75)
				ax_pol_a[1].plot(x2a, g_2a, color=colors[col_ih],
								 label=lbs,
								 linewidth=0.75)

				ax_pol_a[0].set_yticklabels(ax_pol[0].get_yticks(), size=9)
				ax_pol_a[0].set_xticklabels(ax_pol[0].get_xticks(), size=9)
				ax_pol_a[0].yaxis.set_major_formatter(
					FormatStrFormatter("%.1f"))
				ax_pol_a[0].xaxis.set_major_formatter(
					FormatStrFormatter("%.0f"))
				ax_pol_a[1].set_xlabel(r'Time $t$ total resources', fontsize=11)
				ax_pol_a[1].set_ylabel(r'End of time $t$ financial assets', fontsize=11)
				ax_pol_a[0].set_xlabel(r'Time $t$ total resources', fontsize=11)
				ax_pol_a[0].set_ylabel(r'End of time $t$ financial assets', fontsize=11)
				ax_pol_a[1].spines['right'].set_visible(False)
				ax_pol_a[1].spines['top'].set_visible(False)
				ax_pol_a[0].spines['right'].set_visible(False)
				ax_pol_a[0].spines['top'].set_visible(False)
				ax_pol_a[1].set_yticklabels(ax_pol[0].get_yticks(), size=9)
				ax_pol_a[1].set_xticklabels(ax_pol[0].get_xticks(), size=9)
				ax_pol_a[1].yaxis.set_major_formatter(
					FormatStrFormatter("%.1f"))
				ax_pol_a[1].xaxis.set_major_formatter(
					FormatStrFormatter("%.0f"))
				ax_pol_a[1].grid(True)
				ax_pol_a[0].grid(True)
				#]

				ax_val[0].set_yticklabels(ax_val[0].get_yticks(), size=9)
				ax_val[0].set_xticklabels(ax_val[0].get_xticks(), size=9)
				ax_val[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
				ax_val[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
				ax_val[1].set_xlabel(r'Time $t$ total resources', fontsize=11)
				ax_val[1].set_ylabel('Value', fontsize=11)
				ax_val[0].set_xlabel(r'Time $t$ total resources', fontsize=11)
				ax_val[0].set_ylabel('Value', fontsize=11)
				ax_val[1].spines['right'].set_visible(False)
				ax_val[1].spines['top'].set_visible(False)
				ax_val[0].spines['right'].set_visible(False)
				ax_val[0].spines['top'].set_visible(False)
				ax_val[1].set_yticklabels(ax_val[0].get_yticks(), size=9)
				ax_val[1].set_xticklabels(ax_val[0].get_xticks(), size=9)
				ax_val[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
				ax_val[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

				ax_val[0].legend(frameon=False, prop={'size': 10})
				ax_pol[0].legend(frameon=False, prop={'size': 10})

				ax_val[0].set_title(Results1['label'], fontsize=11)
				ax_val[1].set_title(Results2['label'], fontsize=11)

				ax_pol[0].set_title(Results1['label'], fontsize=11)
				ax_pol[1].set_title(Results2['label'], fontsize=11)

				ax_pol_a[0].set_title(Results1['label'], fontsize=11)
				ax_pol_a[1].set_title(Results2['label'], fontsize=11)
				
				ax_pol[1].grid(True)

		


			fig_pol.tight_layout()
			fig_val.tight_layout()

		fig_pol.savefig('../results/plots/durables/policy_adj_housing_{}.png'.format(plot_t))
		fig_val.savefig('../results/plots/durables/value_housing_{}.png'.format(plot_t))
		fig_pol_a.savefig('../results/plots/durables/policy_adj_assets_{}.png'.format(plot_t))

		pl.close()



def plot_grids(adj_ur_grids,cp, term_t = 58):

	for j in list(range(term_t, cp.T)):

		# Test Scan Plots
  
		palette = sns.color_palette("cubehelix", 3)
		palette1 = sns.color_palette("cubehelix", 5)
		color1 = palette[0] 
		color2 = palette[1]
		color3 = palette[2]
		palette[2] = palette1[0]
		colors = palette

		plot_t = j + 1

		sns.set(
		style="white", rc={
		"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

		fig, ax = pl.subplots(2, 2)

		# unrefined grids
		vf_unrefined = adj_ur_grids[plot_t]["EGMGrids"]["vf_unrefined_adj"]
		e_grid_unrefined = adj_ur_grids[plot_t]["EGMGrids"]["endog_grid_unrefined_adj"]
		a_prime_unrefined_adj = adj_ur_grids[plot_t]["EGMGrids"]["a_prime_unrefined_adj"]
		h_prime_unrefined_adj = adj_ur_grids[plot_t]["EGMGrids"]["h_prime_unrefined_adj"]

		# refined non-unform grids
		e_grid_clean = adj_ur_grids[plot_t]["EGMGrids"]["e_grid_clean"]
		vf_clean = adj_ur_grids[plot_t]["EGMGrids"]["vf_clean"]
		hprime_clean = adj_ur_grids[plot_t]["EGMGrids"]["hprime_clean"]
		a_prime_clean = adj_ur_grids[plot_t]["EGMGrids"]["a_prime_clean"]
		sigma_intersect = adj_ur_grids[plot_t]["EGMGrids"]["sigma_intersect"]
		m_intersect = adj_ur_grids[plot_t]["EGMGrids"]["M_intersect"]

		# uniform grids
		#a_prime_adj = adj_ref_grids[plot_t]["a_prime_adj"]
		#h_prime_adj = adj_ref_grids[plot_t]["h_prime_adj"]
		#v_adj = adj_ref_grids[plot_t]["v_adj"]

		ax[0, 0].scatter(e_grid_unrefined, vf_unrefined, color=colors[0],
						 s=15,
						 marker='x',
						 linewidth=0.75)

		ax[0, 0].scatter(e_grid_clean, vf_clean, color=colors[1], s=15,
						 marker='o',
						 linewidth=0.75)

		ax[0, 0].set_ylim(10, 12)

		ax[0, 1].scatter(e_grid_unrefined, h_prime_unrefined_adj, color=colors[1],
						 s=15,
						 marker='x',
						 linewidth=0.75)
		ax[0, 1].scatter(e_grid_clean, hprime_clean, color=colors[2], s=15,
						 marker='o',
						 linewidth=0.75)

		ax[1, 0].scatter(h_prime_unrefined_adj, e_grid_unrefined, color=colors[0],
						 s=15,
						 marker='x',
						 linewidth=0.75)
		ax[1, 0].scatter(hprime_clean, e_grid_clean, color=colors[1],
						 s=15,
						 marker='x',
						 linewidth=0.75)

		ax[1, 1].plot(e_grid_clean, hprime_clean, color=colors[0],
					  linewidth=0.75)

		fig.savefig('../results/plots/durables/scan_test_{}.png'.format(plot_t))

		# FUES-EGM Plots for paper

		pl.close()
		fig, ax = pl.subplots(1, 2)

		sns.set(
			style="white", rc={
				"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

		ax[0].scatter(
			e_grid_unrefined[1:],
			vf_unrefined[1:],
			s=20,
			facecolors='none',
			edgecolors=colors[1])
		ax[0].plot(
			e_grid_clean[1:],
			vf_clean[1:],
			color=colors[2],
			linewidth=1,
			label='Value function')
		ax[0].scatter(
			e_grid_clean[1:],
			vf_clean[1:],
			color=colors[0],
			s=15,
			marker='x',
			linewidth=0.75)

		#ax[0].set_ylim(1.68, 1.72)
		ax[0].set_xlim(30, 40)
		ax[0].set_ylim(2.75,2.9)
		# ax[0].set_xlim(48,56)
		ax[0].set_xlabel(r'Time $t$ total resources', fontsize=11)
		ax[0].set_ylabel('Value', fontsize=11)
		#ax[0].spines['right'].set_visible(False)
		#ax[0].spines['top'].set_visible(False)
		ax[0].legend(frameon=True, prop={'size': 10})
		ax[0].grid(True)
		#ax[0].set_yticklabels(ax[0].get_yticks(), size=9)
		#ax[0].set_xticklabels(ax[0].get_xticks(), size=9)
		# ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
		# ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

		ax[1].scatter(
			e_grid_unrefined,
			h_prime_unrefined_adj,
			s=20,
			facecolor='none',
			edgecolor=colors[1],
			label='EGM points')
		ax[1].scatter(
			e_grid_clean,
			hprime_clean,
			s=20,
			color=colors[0],
			marker='x',
			linewidth=0.75,
			label='Optimal points')
		
		ax[1].plot(e_grid_clean, hprime_clean, color=colors[2], linewidth=1)
		
		if m_intersect is not None:
			ax[1].scatter(
				m_intersect[:,0],
				sigma_intersect[:, 1],
				s=20,
				color='green',
				marker='x',
				linewidth=0.75,
				label='Optimal points')


		# ax[1].set_ylim(20,40)
		#ax[1].set_xlim(44, 54.2)
		ax[1].set_ylim(10, 40)
		ax[1].set_xlim(30, 40)
		ax[1].set_ylabel(r'End of time $t$ housing assets', fontsize=11)
		ax[1].set_xlabel(r'Time $t$ total resources', fontsize=11)
		#ax[1].spines['right'].set_visible(False)
		#ax[1].spines['top'].set_visible(False)
		ax[1].grid(True)
		# ax[1].set_yticklabels(ax[1].get_yticks(), size=9)
		#ax[1].set_xticklabels(ax[1].get_xticks(), size=9)
		# ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
		# ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
		fig.tight_layout()
		ax[1].legend(frameon=True, prop={'size': 10})
		fig.savefig(
			'../results/plots/durables/hous_vf_aprime_all_small_{}.png'.format(plot_t))

		# Plot all EGM points
		pl.close()
		pos = np.where(np.abs(np.diff(hprime_clean)) > 0.4)[0] + 1
		y1 = np.insert(hprime_clean, pos, np.nan)
		x1 = np.insert(e_grid_clean, pos, np.nan)

		fig, ax = pl.subplots(1, 1)
		sns.set(
			style="white", rc={
				"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

		ax.scatter(
			h_prime_unrefined_adj[1:],
			e_grid_unrefined[1:],
			s=20,
			facecolor='none',edgecolor=colors[1], label='EGM points')

		ax.scatter(hprime_clean[1:],
				   e_grid_clean[1:],
				   color=colors[0],
				   s=15,
				   marker='x',
				   linewidth=0.75, label='Optimal points')
		
		if m_intersect is not None:
			ax.scatter(
				sigma_intersect[:, 1],
				m_intersect[:,0],
				s=20,
				color='green',
				marker='x',
				linewidth=0.75,
				label='Optimal points')

		#ax[0].set_ylim(7.75, 8.27)
		#ax[0].set_xlim(30, 40)
		# ax[0].set_ylim(7.3,8)
		# ax.set_xlim(0,75)
		ax.set_xlabel(r'End of time $t$ housing assets', fontsize=11)
		ax.set_ylabel(r'Time $t$ total resources', fontsize=11)
		#ax.spines['right'].set_visible(False)
		#ax.spines['top'].set_visible(False)
		ax.legend(frameon=True, prop={'size': 11})
		ax.grid(True)
		ax.set_xlim(0,30)
		ax.set_ylim(0, 40)
		#ax[0].set_yticklabels(ax[0].get_yticks(), size=9)
		#ax[0].set_xticklabels(ax[0].get_xticks(), size=9)
		# ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
		# ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
		"""
		ax[1].scatter(
			e_grid_unrefined,
			h_prime_unrefined_adj,
			s=20,
			facecolors='none',
			edgecolors='r',
			label='EGM points')
		ax[1].scatter(
			e_grid_clean,
			hprime_clean,
			s=20,
			color='blue',
			marker='x',
			linewidth=0.75,
			label='Optimal points')

		#ax[1].set_ylim(20,40)
		#ax[1].set_xlim(44, 54.2)
		#ax[1].set_ylim(5,40)
		#ax[1].set_xlim(30, 40)
		ax[1].set_ylabel(r'End of time $t$ housing assets', fontsize=11)
		ax[1].set_xlabel('Wealth (t+1)', fontsize=11)
		ax[1].spines['right'].set_visible(False)
		ax[1].spines['top'].set_visible(False)
		# ax[1].set_yticklabels(ax[1].get_yticks(), size=9)
		#ax[1].set_xticklabels(ax[1].get_xticks(), size=9)
		#ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
		#ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))plot_grids
		fig.tight_layout()
		ax[1].legend(frameon=False, prop={'size': 10})
		"""
		fig.savefig('../results/plots/durables/hous_vf_aprime_all_big_{}.png'.format(plot_t))
