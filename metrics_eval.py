import os
import librosa
import preprocess
from mcd import dtw
import mcd.metrics_fast as mt
import argparse

def pitch_cadence_distance(f0, f1, silence_dist=10):
    mem = {}
    def helper(s0, s1):
        if (s0, s1) in mem:
            return mem[s0, s1]
        if s0 == len(f0):
            return sum(f1[s1:])
        if s1 == len(f1):
            return sum(f0[s0:])
        change = helper(s0+1, s1+1) + abs(f0[s0] - f1[s1])
        insert0 = helper(s0, s1+1) + max(silence_dist, f1[s1])
        insert1 = helper(s0+1, s1) + max(silence_dist, f0[s0])
        min_dist = min([change, insert0, insert1])
        mem[s0, s1] = min_dist
        return min_dist
    return helper(0,0) / (len(f0)+len(f1)) * 2

def MCD(sp1, sp2, costFn):
    min_cost, path = dtw.dtw(sp1, sp2, costFn)
    frames = len(sp1)
    return min_cost, frames, path

def process_file(filePath):
    num_mcep = 24
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128
    wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
    wav = preprocess.wav_padding(wav=wav,
                                sr=sampling_rate,
                                frame_period=frame_period,
                                multiple=4)
    f0, timeaxis, sp, ap = preprocess.world_decompose(
            wav=wav, fs=sampling_rate, frame_period=frame_period)
    coded_sp = preprocess.world_encode_spectral_envelop(
            sp=sp, fs=sampling_rate, dim=num_mcep)
    return coded_sp, f0


def mean_metrics_for_dirs(A_dir, B_dir):
    costFn = mt.logSpecDbDist

    total_mcd = 0.0
    total_pcd = 0.0
    num_wav = 0

    for file in os.listdir(A_dir):
        print("processing ", file)
        filePathA = os.path.join(A_dir, file)
        filePathB = os.path.join(B_dir, file)
        sp1, f01 = process_file(filePathA)
        sp2, f02 = process_file(filePathB)

        mcd, _, _ = MCD(sp1, sp2, costFn)
        mean_mcd = mcd / (len(sp1) + len(sp2)) * 2
        total_mcd += mean_mcd

#        pcd = pitch_cadence_distance(f01, f02)
        pcd = 0.0 # disabled for debug
        total_pcd += pcd

        num_wav += 1

    return total_mcd / num_wav, total_pcd / num_wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generated-dir", type=str, required=True)
    parser.add_argument("-t", "--target-dir", type=str, required=True)
    args = parser.parse_args()

    print(mean_metrics_for_dirs(args.generated_dir, args.target_dir))

