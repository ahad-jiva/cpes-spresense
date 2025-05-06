import tensorflow as tf
import binascii

train_ds = tf.data.Dataset.load('train_data')

for images, labels in train_ds.take(1):
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(numpy_images).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_saved_model('mymodel')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS
]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

# save to disk
with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model_quant)

import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Display Input Details
print("Input Details with Quantization Parameters:")
for input_detail in input_details:
    print(f"  Name: {input_detail['name']}")
    print(f"  Shape: {input_detail['shape']}")
    print(f"  Data Type: {input_detail['dtype']}")
    print(f"  Quantization Parameters: {input_detail['quantization']}")
    print(f"  Quantization Scale: {input_detail['quantization_parameters']['scales']}")
    print(f"  Quantization Zero Points: {input_detail['quantization_parameters']['zero_points']}\n")

# Display Output Details
print("\nOutput Details with Quantization Parameters:")
for output_detail in output_details:
    print(f"  Name: {output_detail['name']}")
    print(f"  Shape: {output_detail['shape']}")
    print(f"  Data Type: {output_detail['dtype']}")
    print(f"  Quantization Parameters: {output_detail['quantization']}")
    print(f"  Quantization Scale: {output_detail['quantization_parameters']['scales']}")
    print(f"  Quantization Zero Points: {output_detail['quantization_parameters']['zero_points']}\n")

def convert_to_c_array(bytes_data) -> str:
    hexstr = binascii.hexlify(bytes_data).decode("UTF-8").upper()
    array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
    array = [array[i:i+10] for i in range(0, len(array), 10)]
    return ",\n  ".join([", ".join(e) for e in array])

# Read the TFLite model
with open('model_int8.tflite', 'rb') as f:
    tflite_binary = f.read()

ascii_bytes = convert_to_c_array(tflite_binary)
header_file_content = (
    "const unsigned char model_tflite[] = {\n  "
    + ascii_bytes
    + "\n};\nunsigned int model_tflite_len = "
    + str(len(tflite_binary)) + ";"
)

# Write to .h file
with open("model.h", "w") as f:
    f.write(header_file_content)