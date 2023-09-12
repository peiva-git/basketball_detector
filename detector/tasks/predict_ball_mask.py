import os.path
import pathlib

import onnx
from vidgear.gears import CamGear

import cv2 as cv
import onnxruntime as ort
import numpy as np

TEST_DATA_DIR = '/mnt/DATA/tesi/dataset/dataset_paddleseg/images/'

if __name__ == '__main__':
    session = ort.InferenceSession(
        '/home/peiva/ppliteSeg/inference_model.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    stream = CamGear(source='/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste/stagione_2019-20_legabasket/pallacanestro_trieste-dolomiti_energia_trentino/final_cut.mp4').start()
    index = 1
    while True:
        frame = stream.read()
        if frame is None:
            break
        frame_resized = cv.resize(frame, (2048, 1024))
        frame_channels_first = np.moveaxis(cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB), source=-1, destination=0)
        batch = np.expand_dims(frame_channels_first, axis=0)
        input_name = session.get_inputs()[0].name
        print(f'Input name: {input_name}')

        output = session.run([], {input_name: batch.astype(np.float32)})
        mask_batch = np.asarray(output, dtype=np.uint8)
        mask_channels_first = np.squeeze(mask_batch, axis=0)
        mask = np.moveaxis(mask_channels_first, source=0, destination=-1).squeeze()
        # mask_rescaled = mask * 255
        color = np.array([0, 255, 0], dtype=np.uint8)
        masked_image = np.where(mask[..., None], color, frame_resized)
        frame_with_overlay = cv.addWeighted(frame_resized, 0.8, masked_image, 0.2, 0)
        # cv.imshow('Output', frame_with_overlay)
        cv.imwrite(f'/home/peiva/experiments/test_videos/overlay{index}.png', frame_with_overlay)
        index += 1

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
    stream.stop()
