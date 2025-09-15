# General imports
import numpy as np
import matplotlib.pyplot as plt

# BLonD imports
import blond.llrf.barrier_bucket as bbuck
import blond.beam.beam as bbeam
import blond.input_parameters.ring as bring
import blond.input_parameters.rf_parameters as brf
import blond.trackers.tracker as btrack

#%% Barrier definition

t_rev = 2E-6
pos_centered = t_rev/2
pos_edges = t_rev

barrier_width = 0.3E-6
barrier_amplitude = 5E3

##################################################################
# Construct two barrier waveforms, one centered in the revolution
# period, the other offset to the edges.
##################################################################
bbgen_centered = bbuck.BarrierGenerator(pos_centered, barrier_width,
                                        barrier_amplitude)
bbgen_edges = bbuck.BarrierGenerator(pos_edges, barrier_width,
                                     barrier_amplitude)

barrier_bins = np.linspace(0, t_rev, 1000)
centered_barrier = bbgen_centered.waveform_at_time(0, barrier_bins)
edges_barrier = bbgen_edges.waveform_at_time(0, barrier_bins)

plt.plot(barrier_bins*1E9, centered_barrier/1E3, color='blue', lw=3,
         label='Centered Barrier')
plt.plot(barrier_bins*1E9, edges_barrier/1E3, color='red', lw=3,
         label='Edges Barrier')
plt.axhline(barrier_amplitude/1E3)
plt.legend(fontsize=13)
plt.xlabel("dt (ns)", fontsize=13)
plt.ylabel("Amplitude (kV)", fontsize=13)
plt.title("Ideal barrier comparison", fontsize=14)
plt.tight_layout()
plt.show()

#%% Barrier harmonic series

######################################################################
# Convert the barriers to a harmonic series with low pass filtering
# to prevent excessive ringing either side of the barrier.
######################################################################
barrier_harmonics = list(range(1, 26))

centered_terms = bbgen_centered.for_rf_station(times = [0],
                                               t_rev = [t_rev],
                                               harmonics = barrier_harmonics,
                                               m=2)
offset_terms = bbgen_edges.for_rf_station(times = [0],
                                          t_rev = [t_rev],
                                          harmonics = barrier_harmonics,
                                          m=2)

print("Centered barrier, contribution per harmonic")
for i in range(len(barrier_harmonics)):
    print(f"h={centered_terms[0][i]}, V={centered_terms[1][i][1, 0]/1E3:.2f} kV "
          + f"phi={centered_terms[2][i][1, 0]:.2f} rad")

print("\n\n")

print("Offset barrier, contribution per harmonic")
for i in range(len(barrier_harmonics)):
    print(f"h={offset_terms[0][i]}, V={offset_terms[1][i][1, 0]/1E3:.2f} kV "
          + f"phi={offset_terms[2][i][1, 0]:.2f} rad")

centered_amplitudes = []
centered_phases = []
for i in range(len(barrier_harmonics)):
    centered_amplitudes.append(centered_terms[1][i][1, 0])
    centered_phases.append(centered_terms[2][i][1, 0])

offset_amplitudes = []
offset_phases = []
for i in range(len(barrier_harmonics)):
    offset_amplitudes.append(offset_terms[1][i][1, 0])
    offset_phases.append(offset_terms[2][i][1, 0])

centered_waveform = bbuck.harmonics_to_waveform(barrier_bins,
                                                centered_terms[0],
                                                centered_amplitudes,
                                                centered_phases)

offset_waveform = bbuck.harmonics_to_waveform(barrier_bins,
                                              offset_terms[0],
                                              offset_amplitudes,
                                              offset_phases)

plt.plot(barrier_bins*1E9, centered_waveform/1E3, color='blue', lw=3,
         label='Centered Barrier')
plt.plot(barrier_bins*1E9, offset_waveform/1E3, color='red', lw=3,
         label='Edges Barrier')
plt.axhline(barrier_amplitude/1E3)
plt.legend(fontsize=13)
plt.xlabel("dt (ns)", fontsize=13)
plt.ylabel("Amplitude (kV)", fontsize=13)
plt.title("Reconstructed barrier comparison", fontsize=14)
plt.tight_layout()
plt.show()

#%% Ideal vs reconstructed comparison

plt.plot(barrier_bins*1E9, centered_barrier/1E3, color='blue', lw=3,
         linestyle = '--', label='Centered Barrier - Ideal')
plt.plot(barrier_bins*1E9, centered_waveform/1E3, color='blue', lw=3,
         label='Centered Barrier - Reconstructed')

plt.plot(barrier_bins*1E9, edges_barrier/1E3, color='red', lw=3,
         linestyle = '--', label='Edges Barrier - Ideal')
plt.plot(barrier_bins*1E9, offset_waveform/1E3, color='red', lw=3,
         label='Edges Barrier - Reconstructed')
plt.axhline(barrier_amplitude/1E3)
plt.legend(fontsize=13)
plt.xlabel("dt (ns)", fontsize=13)
plt.ylabel("Amplitude (kV)", fontsize=13)
plt.title("Ideal vs reconstructed barrier", fontsize=14)
plt.tight_layout()
plt.show()


#%% Prepare tracking

# Ring
rad = 100
circ = 2*np.pi * rad
gamma_t = 6.1
mom_comp = 1 / (gamma_t**2)
momentum = 14E9

ring = bring.Ring(circ, mom_comp, momentum, bbeam.Proton(), 10000)

# RF station
b_harm, b_volt, b_phase = offset_terms

rf = brf.RFStation(ring, b_harm, tuple(b_volt), tuple(b_phase), len(b_harm))

# Beam
n_macro = int(1E5)
beam = bbeam.Beam(ring, n_macro, intensity = 0)
beam.dt[:] = np.random.uniform(0.3E-6, ring.t_rev[0] - 0.3E-6, n_macro)
beam.dE[:] = np.random.uniform(-15E6, 15E6, n_macro)

# Tracker
long_tracker = btrack.RingAndRFTracker(rf, beam)
full_tracker = btrack.FullRingAndRF([long_tracker])

#%% Track

for i in range(ring.n_turns):
    full_tracker.track()
    if i % 1000 == 0:
        plt.scatter(beam.dt[::10]*1E9, beam.dE[::10]/1E6)
        plt.gca().twinx().plot(barrier_bins*1E9, offset_waveform/1E3, color='red')
        plt.show()
