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
    char_codes = 'lrudswx'
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
                    if chr(key_code) in char_codes:
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


@router_cmd.command()
@click.option('--path2network')
def inference(path2network):
    try:
        drone_api = Tello()
        drone_api.connect()
        battery = drone_api.get_battery()
        logger.debug(f'battrery : {battery:03d} %')
        
        marker0 = 0
        marker1 = 0 

        W, H = 640, 480

        screen0 = '000'
        create_window(screen0, (W, H), (100, 100))

        screen1 = '001'
        create_window(screen1, (W, H), (800, 100))
        
        device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    
        network = th.load(path2network)
        network.to(device)
        network.eval()
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        mp_builder = mp.solutions.hands
        mp_builder_config = {
            'max_num_hands': 2,
            'model_complexity': 0,
            'min_tracking_confidence': 0.5,
            'min_detection_confidence': 0.5
        }

        command_counter = {
            'DOWN': 0,
            'UP': 0,
            'LEFT': 0,
            'RIGHT': 0,
            'TAKEOFF': 0,
            'LAND': 0,
            'STOP': 0
        }

        command_status = {
            'DOWN': 0,
            'UP': 0,
            'LEFT': 0,
            'RIGHT': 0,
            'TAKEOFF': 0,
            'LAND': 0
        }
        nb_validation = 2  # a command is valid if it was detected twice continuouly 
        char_codes = 'lrudswx'
        commands = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'STOP', 'TAKEOFF', 'LAND']
        map_code2command = dict(zip(char_codes, commands))
        with mp_builder.Hands(**mp_builder_config) as detector:
            capture = cv2.VideoCapture(0)
            keep_capture = True 
            while keep_capture:
                logger.debug(f'drone height : {drone_api.get_height():03d} cm')
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
                        draw_landmarks(mp_drawing, mp_drawing_styles, mp_builder, bgr_frame, hand)
                        points = [ [pnt.x, pnt.y] for pnt in hand.landmark ]
                        points = np.asarray(points, dtype=np.float16)
                        rescaled_points = points * scaler 
                        distance_matrix = build_matrix(rescaled_points)
                        distance_matrix /= np.max(distance_matrix)  # normalize between 0 and 1 
                        adjacency_matrix = (distance_matrix * 255).astype('uint8')
                        
                        cv2.imshow(screen1, cv2.resize(adjacency_matrix, (W, H)))

                        input_batch = th.tensor(np.ravel(distance_matrix)).float()[None, ...]
                        output = network(input_batch.to(device))
                        output = th.squeeze(output).cpu()
                        candidate = th.argmax(output)
                        predicted_command = map_code2command[char_codes[candidate]]
                        if command_counter[predicted_command] < nb_validation:
                            command_counter[predicted_command] += 1
                        else:
                            if predicted_command != 'LAND':
                                command_counter[predicted_command] = 0
                                command_counter['LAND'] = 0 
                            else:
                                command_counter[predicted_command] += 1

                            logger.success(f'COMMAND: {predicted_command:<30} will be executed')
                            if command_status['TAKEOFF']:
                                if predicted_command == 'DOWN':
                                    if command_status[predicted_command] == 0 and drone_api.get_height() >= 0:
                                        drone_api.move_down(20)
                                        command_status[predicted_command] = 1

                                if predicted_command == 'UP':
                                    if command_status[predicted_command] == 0 and drone_api.get_height() <= 20:
                                        drone_api.move_up(20)
                                        command_status[predicted_command] = 1

                                if predicted_command == 'LEFT':
                                    if command_status[predicted_command] == 0:
                                        drone_api.move_left(20)
                                        command_status[predicted_command] = 1

                                if predicted_command == 'RIGHT':
                                    if command_status[predicted_command] == 0:
                                        drone_api.move_right(20)
                                        command_status[predicted_command] = 1
                                
                                if predicted_command == 'STOP':
                                    # do not reset land and takeoff status
                                    command_counter['LAND'] = 0 
                                    for key in ['LEFT', 'RIGHT', 'UP', 'DOWN', 'LAND']:
                                        command_status[key] = 0 

                            if predicted_command == 'TAKEOFF':
                                if command_status[predicted_command] == 0:
                                    drone_api.takeoff()
                                command_status[predicted_command] = 1
                            
                            if predicted_command == 'LAND' and command_counter[predicted_command] == 10 * nb_validation + 1:
                                command_status[predicted_command] = 1
                                break 
                    cv2.imshow(screen0, bgr_frame)
            # end while loop 
            cv2.destroyAllWindows()
            logger.debug('wait drone api to land')
        # end context manager 
    except KeyboardInterrupt as e:
        if drone_api is not None:
            drone_api.land()
    except Exception as e:
        logger.error(e)
        if drone_api is not None:
            drone_api.land()
    finally:
        drone_api.land()
        while drone_api.get_height() > 0:
            sleep(1)
        logger.success('all ressources were removed')

if __name__ == '__main__':
    router_cmd(obj={})