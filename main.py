import click 
import cv2
import pickle as pk 

import numpy as np 
import torch as th 
import torch.nn as nn 
import mediapipe as mp 

import operator as op
import itertools as it, functools as ft 

from torch.utils.data import TensorDataset, DataLoader
from djitellopy import Tello
from time import sleep 

from log import logger 
from model import MLP_Model
from strategies import * 

from google.protobuf.json_format import MessageToDict

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def router_cmd(ctx, debug):
    if ctx.invoked_subcommand is not None:
        logger.debug(f'{ctx.invoked_subcommand} subcommand was called')
    ctx.obj['debug'] = debug 
    if debug:
        logger.debug('debug-mode was activated')


@router_cmd.command()
@click.option('--path2data', help='path where the features will be stored', type=click.File(mode='wb'))
@click.pass_context
def prepare_data(ctx, path2data):
    mp_builder = mp.solutions.hands
    mp_builder_config = {
        'max_num_hands': 1,
        'model_complexity': 0,
        'min_tracking_confidence': 0.5,
        'min_detection_confidence': 0.5
    }
    char_codes = 'lruds'
    with mp_builder.Hands(**mp_builder_config) as detector:
        capture = cv2.VideoCapture(0)
        keep_capture = True 
        accumulator = []
        while keep_capture:
            key_code = cv2.waitKey(25) & 0xFF 
            cap_status, bgr_frame = capture.read()
            keep_capture = key_code != 27  # hit [escape] to end the loop 
            if cap_status and keep_capture:
                h, w, _ = bgr_frame.shape 
                scaler = np.array([w, h])
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                response = detector.process(rgb_frame)
                response_data = response.multi_hand_landmarks 
                if response_data is not None:
                    hand = response_data[0]
                    points = [ [pnt.x, pnt.y] for pnt in hand.landmark ]
                    points = np.asarray(points, dtype=np.float16)
                    rescaled_points = points * scaler 
                    distance_matrix = build_matrix(rescaled_points)
                    distance_matrix /= np.max(distance_matrix)  # normalize between 0 and 1 
                    pixels = (distance_matrix * 255).astype('uint8')
                    cv2.imshow('001', cv2.resize(pixels, (640, 480)))
                    if chr(key_code) in 'lruds':
                        accumulator.append((np.ravel(distance_matrix), char_codes.index(chr(key_code))))
                        print(key_code) 
                
                cv2.imshow('000', bgr_frame)
        # end while loop 

        pk.dump(accumulator, path2data)
    # end context manager 


@router_cmd.command()
@click.option('--path2data', help='path where landmarks where stored')
@click.option('--nb_epochs', type=int, help='number of epochs')
@click.option('--batch_size', type=int, help='size of input batch')
def train(path2data, nb_epochs, batch_size):
    with open(path2data, 'rb') as fp:
        landmarks_labels = pk.load(fp)
    landmarks, labels = list(zip(*landmarks_labels))
    landmarks = th.tensor(landmarks).float()
    labels = th.tensor(labels).long()

    dataset = TensorDataset(landmarks, labels)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    nb_classes = len(np.unique(labels.numpy()))

    nb_items = len(dataset)
    
    logger.debug(f'dataset size: {len(dataset):05d} | nb classes : {nb_classes:03d}')
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    model = MLP_Model([441, 128, 64, nb_classes], [1, 1, 0], [1, 1, 0])
    model.to(device)
    model.train()

    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(nb_epochs):
        item_cnt = 0 
        for X, Y in dataloader:
            item_cnt += X.shape[0]  # size of batch 
            X = X.to(device)
            Y = Y.to(device)
            P = model(X)

            optimizer.zero_grad()
            E = criterion(P, Y)
            E.backward()
            optimizer.step()

            V = E.cpu().item()

            logger.debug(f'[{epoch:03d}/{nb_epochs:03d}]:[{item_cnt:05d}/{nb_items:05d}] >> loss : {V:07.3f}')
    
    th.save(model.cpu(), 'network.th')
    logger.success('the model was saved ...!')


if __name__ == '__main__':
    router_cmd(obj={})