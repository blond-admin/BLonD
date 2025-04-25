import pickle as pkl
import matplotlib.pyplot as plt
n_section_max = 3
dictresults = []
for i in range(n_section_max):
    filename = f'summary_characteristics_{i+1}_rf_sections_{'ttbar'}_mode'
    dictresults.append(pkl.load(open(filename, "rb")))

directory = 'comparison_multi_rf_sections'
option_summary =directory

figpos, axpos = plt.subplots()
axpos.set_title(f'Average bunch position [ns]')

axpos.set(xlabel='turn', ylabel = 'Bunch position [ns]', ylim = [0.3, 0.7])
for i in range(n_section_max):
    axpos.plot(dictresults[i]['tracking_bpos_ns'], label = f'n_sections = {i+1}')
axpos.plot(dictresults[0]['expected_bpos_ns'], ls = '--', lw = 2, label='expected')
axpos.legend()
plt.savefig(directory+'/bunch_position'+option_summary)
plt.close()

figbl, axbl = plt.subplots()
axbl.set(xlabel='turn', ylabel = 'Bunch length [mm]', ylim = [1.5, 4])
axbl.set_title(f'RMS bunch length [mm]')
for i in range(n_section_max):
    axbl.plot(dictresults[i]['tracking_bl_mm'], label = f'n_sections = {i+1}')
axbl.plot(dictresults[0]['expected_bl_mm'], ls = '--', lw = 2,label = 'expected')
axbl.legend()
plt.savefig(directory+'/bunch_length'+option_summary)
plt.close()

fig, ax = plt.subplots()
for i in range(n_section_max):
    ax.plot(dictresults[i]['tracking_sigmaE_perc'], label = f'n_sections = {i+1}')
ax.plot(dictresults[0]['expected_sigmaE_perc'],  ls = '--', lw = 2,label = 'expected')
ax.legend()
ax.set(xlabel='turn', ylabel = 'Energy spread [%]', ylim = [0.03, 0.17])
ax.set_title(f'RMS energy spread [%]')
plt.savefig(directory+'/energy_spread'+option_summary)
plt.close()