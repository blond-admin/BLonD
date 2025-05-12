from matplotlib import pyplot as plt
import pickle as pkl
import seaborn as sns
import imageio
import os
import numpy as np

opmode = 'W'

folder_paths = [
    '/Users/lvalle/PycharmProjects/BLonD/Simulations_ramps/W_mode/data_figs/',
]

gif_paths = [
    '/Users/lvalle/PycharmProjects/BLonD/Simulations_ramps/W_mode/gif_path/animated_ramp.gif',
]


n=0

get_images=True
# Folder to save frames
folder_name = opmode + '_mode/frames'
os.makedirs("frames", exist_ok=True)
filenames = []

if get_images:
    for i in range(len(os.listdir(folder_paths[0]))):
        file = f'{folder_paths[n]}beam_{i+1}.pkl'
        file_HB = f'{folder_paths[n]}sep_{i+1}.npy'
        file_HB_X = f'{folder_paths[n]}sep_{i + 1}_X.npy'
        file_HB_Y = f'{folder_paths[n]}sep_{i + 1}_Y.npy'
        df_beam = pkl.load(open(file, "rb"))
        Z = np.load(file_HB)
        X = np.load(file_HB_X)
        Y = np.load(file_HB_Y)
        dt = df_beam.dt[0:10000] * 1e9
        dE = df_beam.dE[0:10000] * 1e-9

        plt.figure(figsize=(6, 5))
        plt.title(f'WW mode\n Turn: {i}')
        plt.contour(X, Y, Z, HDE, colors=['red'])
        sns.scatterplot(data = [dt,dE], x='t [ns]', y='DE [GeV]', bins=100, marginal_kws=dict(bins=100, fill=False))
        #sns.set(xlabel = 't [ns]', ylabel = 'DE [GeV]', xlim = [0,1.25])
        sns.kdeplot(data = Z, x=X, y=Y)
        frame_path = f"frames/frame_{i:03d}.png"

        plt.savefig(frame_path)
        plt.close()
        filenames.append(frame_path)
        print(f"Iteration {i} done")
        plt.close()
else :
    for i in range(len(os.listdir(folder_paths[0]))):
        frame_path = f"frames/frame_{i:03d}.png"
        filenames.append(frame_path)

if filenames:
    with imageio.get_writer(gif_paths[n], mode='I', duration=0.2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            print(f"Added {filename} to GIF")

    print("GIF saved as beam_evolution.gif.")
else:
    print("No frames to create a GIF.")

# # Clean up frames
for filename in filenames:
    if os.path.exists(filename):
        os.remove(filename)
if os.path.exists("frames") and not os.listdir("frames"):
    os.rmdir("frames")


    print(f"Saved frame {i:03d} at {frame_path}")

print("Temporary frames deleted.")