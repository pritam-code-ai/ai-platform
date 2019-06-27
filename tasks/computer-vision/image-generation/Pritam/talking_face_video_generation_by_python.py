#!/usr/bin/env python
# coding: utf-8



import librosa
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import os, shutil, subprocess
from keras import backend as K
from keras.models import Model, Sequential, load_model
from tqdm import tqdm
import dlib
from matplotlib import transforms
import cv2
import math
import copy
import mlflow 
import mlflow.keras as mlflow_keras_model
from mlflow.tracking import MlflowClient 
import argparse



font = {'size'   : 0.001}
mpl.rc('font', **font)




# Lookup tables for drawing lines between points

Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57],          [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66],          [66, 67], [67, 60]]


Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33],         [33, 34], [34, 35], [27, 31], [27, 35]]





leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]




leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]





other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],          [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],          [12, 13], [13, 14], [14, 15], [15, 16]]



faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other



class faceNormalizer(object):
    # Credits: http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
    w = 600
    h = 600

    def __init__(self, w = 600, h = 600):
        self.w = w
        self.h = h

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)
      
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        
        return tform

    def tformFlmarks(self, flmark, tform):
        transformed = np.reshape(np.array(flmark), (68, 1, 2))           
        transformed = cv2.transform(transformed, tform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def alignEyePointsV2(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
    
        for i, lmark in enumerate(alignedSeq):
            curLmark = alignedSeq[i,:,:]
            eyecornerSrc  = [ (curLmark[36, 0], curLmark[36, 1]), (curLmark[45, 0], curLmark[45, 1]) ]
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst);
            alignedSeq[i,:,:] = self.tformFlmarks(lmark, tform)

        return alignedSeq




def write_video_wpts_wsound(frames, sound, fs, path, fname, xLim, yLim):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_file.mp4'))
    except:
        print('Exp')

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))
    print(frames.shape)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='talking face generation', artist='Matplotlib',
                    comment='deep learning project')
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    rect = (0, 0, 600, 600)
    
    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print(lookup)
    else:
        lookup = faceLmarkLookup

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(lookup))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -y -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_file.mp4'
    subprocess.call(cmd, shell=True) 
    print('Mixing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))




def melSpectra(y, sr, wsize, hsize):
    cnst = 1+(int(sr*wsize)/2)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst

    melspec = np.log(1e-16+librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec




def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx



def process_and_generate_video_data():
    
    
    output_path = "video_output_data"
    num_features_Y = 136
    num_frames = 75
    wsize = 0.04
    hsize = wsize
    fs = 44100
    trainDelay = 1
    ctxWin = 3
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    


    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained_model_path_as_string", type=str, default = "D80_C3.h5", help='input a string')
    parser.add_argument("--input_audio_file_path_as_string", type=str, default = "audio_input_data/summary.flac", help='input a string')

    args = parser.parse_args()

    print(args.pre_trained_model_path_as_string)
    print(args.input_audio_file_path_as_string)


    model = load_model(args.pre_trained_model_path_as_string)
    
    test_file = args.input_audio_file_path_as_string
    
    # Used for padding zeros to first and second temporal differences
    zeroVecD = np.zeros((1, 64), dtype=np.longdouble)
    zeroVecDD = np.zeros((2, 64), dtype=np.longdouble)
    
    # Load speech and extract features
    sound, sr = librosa.load(test_file, sr=fs)
    melFrames = np.transpose(melSpectra(sound, sr, wsize, hsize))
    melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
    melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
    
    features = np.concatenate((melDelta, melDDelta), axis=1)
    features = addContext(features, ctxWin)
    features = np.reshape(features, (1, features.shape[0], features.shape[1]))

    upper_limit = features.shape[1]
    lower = 0
    generated = np.zeros((0, num_features_Y))
    
    # Generates face landmarks one-by-one

    for i in tqdm(range(upper_limit)):
        cur_features = np.zeros((1, num_frames, features.shape[2]))
        if i+1 > 75:
            lower = i+1-75
        cur_features[:,-i-1:,:] = features[:,lower:i+1,:]

        pred = model.predict(cur_features)
        generated = np.append(generated, np.reshape(pred[0,-1,:], (1, num_features_Y)), axis=0)

        # Shift the array to remove the delay
    generated = generated[trainDelay:, :]
    tmp = generated[-1:, :]
    for _ in range(trainDelay):
        generated = np.append(generated, tmp, axis=0)

    if len(generated.shape) < 3:
        generated = np.reshape(generated, (generated.shape[0], int(generated.shape[1]/2), 2))
        
    fnorm = faceNormalizer()
    generated = fnorm.alignEyePointsV2(600*generated) / 600.0 
    write_video_wpts_wsound(generated, sound, fs, output_path, 'talking_face_generated', [0, 1], [0, 1])

    return model



if __name__ == "__main__":
    
    model = process_and_generate_video_data()
    
    talking_face_video = "video_output_data/talking_face_generated_file.mp4"
    
    file_path = talking_face_video
    
    vid = cv2.VideoCapture(file_path)
    
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    

    if not os.path.isdir("client_uri"):
        os.makedirs("client_uri")
    
    mlflow.tracking.set_tracking_uri("client_uri")
    
    mlflow.tracking.get_tracking_uri()
    

    client = MlflowClient()
    
    experiments = client.create_experiment("talking_face_generation")
    
    run = client.create_run(experiments[0])
    
    client.log_metric(run.info.run_id,"resolution_data_height_of_video",height)
    
    client.log_metric(run.info.run_id,"resolution_data_width_of_video",width)
    
    client.log_artifact(run.info.run_id, "video_output_data/talking_face_generated_file.mp4")
    
    client.log_artifact(run.info.run_id, "audio_input_data/summary.flac")
    
    client.set_tag(run.info.run_id, "mlflow_tracking_code", "mlflow_tracking_code")
    
    mlflow_keras_model.log_model(model,"model_1")
    
    client.get_metric_history(run.info.run_id, "resolution_data_height_of_video")
    
    client.get_metric_history(run.info.run_id, "resolution_data_width_of_video")
    
    client.list_artifacts(run.info.run_id)
    
    client.list_experiments()
    
    client.list_run_infos("0")
    
    client.get_run(run.info.run_id)
    
    client.get_experiment_by_name("talking_face_generation")
    
    client.get_experiment("0")
    
    #client.search_runs(["0",1,"2"], max_results=3)
    
    client.rename_experiment("0", "talking_face_generation_renamed")
        
    client.get_experiment("0")
    
    client.set_terminated(run.info.run_id)
    

