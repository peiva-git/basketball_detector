import os
import pathlib

import onnx
from vidgear.gears import CamGear

import cv2 as cv
import onnxruntime as ort
import numpy as np
import fastdeploy as fd

TEST_DATA_DIR = '/mnt/DATA/tesi/dataset/dataset_paddleseg/images/'
MODEL_PATH = '/home/peiva/ppliteSeg/inference_model.onnx'
# '/home/ubuntu/PaddleSeg/output/inference_model.onnx'

if __name__ == '__main__':
    print('Building model...')
    option = fd.RuntimeOption()
    option.use_gpu()
    model_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdmodel')
    params_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdiparams')
    config_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'deploy.yaml')
    model = fd.vision.segmentation.PaddleSegModel(
        model_file, params_file, config_file, runtime_option=option
    )
    print('Reading video...')
    stream = CamGear(
        #source='/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste/stagione_2019-20_legabasket/pallacanestro_trieste-dolomiti_energia_trentino/final_cut.mp4'
        source='/home/ubuntu/test_video.mp4'
    ).start()
    index = 1
    while True:
        frame = stream.read()
        if frame is None:
            break
        frame_resized = cv.resize(frame, (2048, 1024))
        # frame_channels_first = np.moveaxis(cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB), source=-1, destination=0)
        # batch = np.expand_dims(frame_channels_first, axis=0)
        # input_name = runtime.get_input_info(0).name
        # print(f'Input name: {input_name}')
        #
        # output = runtime.infer({
        #     input_name: batch.astype(np.float32)
        # })
        # mask_channels_first = np.asarray(output[0], dtype=np.uint8)
        # mask = np.moveaxis(mask_channels_first, source=0, destination=-1).squeeze()
        # # mask_rescaled = mask * 255
        # color = np.array([0, 255, 0], dtype=np.uint8)
        # masked_image = np.where(mask[..., None], color, frame_resized)
        # frame_with_overlay = cv.addWeighted(frame_resized, 0.8, masked_image, 0.2, 0)
        #cv.imshow('Output', frame_with_overlay)
        #cv.imwrite(f'/home/peiva/experiments/test_videos/overlay{index}.png', frame_with_overlay)
        print(f'Predicting frame {index}...')
        result = model.predict(frame_resized)
        vis_im = fd.vision.vis_segmentation(frame_resized, result, weight=0.5)
        print(f'Writing overlay {index} to disk...')
        cv.imwrite(f'/home/ubuntu/results/overlay{index}.png', vis_im)
        index += 1

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
    stream.stop()
