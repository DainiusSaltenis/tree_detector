def export_to_onnx(models, args=None):
    import os
    os.environ['TF_KERAS'] = '1'

    import keras2onnx
    import onnxruntime

    model = models['debug_model']
    print('Loading model, this may take a second...')
    model.load_weights(args.snapshot, by_name=True)
    onnx_model = keras2onnx.convert_keras(model, args.model_name)
    temp_model_file = 'model.onnx'
    keras2onnx.save_model(onnx_model, temp_model_file)

    # check if session can be started and model was converted correctly
    sess = onnxruntime.InferenceSession(temp_model_file)