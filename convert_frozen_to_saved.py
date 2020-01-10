# from tensorflow.python.saved_model import signature_constants
# from tensorflow.python.saved_model import tag_constants
# import tensorflow as tf

# export_dir = 'saved_model'
# graph_pb = 'model/saved_model.pb'

# builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
 
# with tf.gfile.GFile(graph_pb, "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

# sigs = {}

# with tf.Session(graph=tf.Graph()) as sess:
#     # name="" is important to ensure we don't get spurious prefixing
#     tf.import_graph_def(graph_def, name="")
#     g = tf.get_default_graph()
#     inp = g.get_tensor_by_name("ReadFile/filename:0")
#     out = g.get_tensor_by_name("Squeeze_1:0")

#     sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
#         tf.saved_model.signature_def_utils.predict_signature_def(
#             {"in": inp}, {"out": out})

#     builder.add_meta_graph_and_variables(sess,
#                                          [tag_constants.SERVING],
#                                          signature_def_map=sigs)

from tensorflow.python.platform import gfile
import tensorflow as tf
import os,sys


config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

with gfile.FastGFile('model/wave.pb', 'rb') as f: 
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') 
sess.run(tf.global_variables_initializer())

input_img = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name('Const:0'))
output = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name('Squeeze_1:0'))
print(output)

export_path_base = "saved_model/wave"
export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes('1'))

# Export model with signature
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'inputs': input_img},
        outputs={'outputs': output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature},main_op=tf.tables_initializer())

builder.save()
