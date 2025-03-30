import tensorflow as tf

# 如果是tensorflow 1.x, 需要打开eager模式
tf.enable_eager_execution()

def get_inofs_from_feature_dict(feature_dict):
    """
    这个函数用于解析feature_dict中的每一个Feature.
    每一个Feature均包含bytes_list, float_list, int64_list. 需要解析.
    只能解析固定长度的Feature. 因为只解析DataSet中的第1个.
    Args:
        feature_dict: key是string, value是tf.core.example.feature_pb2.Feature
    """
    feature_values = dict()
    feature_lens = dict()
    feature_types = dict()

    for feature, value in feature_dict.items():
        bytes_list = value.bytes_list.value
        float_list = value.float_list.value
        int64_list = value.int64_list.value

        feature_len = 0
        feature_type = "unknown"
        feature_value = None
        for i, each_type_value in enumerate((bytes_list, float_list, int64_list)):
            each_type_len = len(each_type_value)
            if each_type_len > 0:
                if feature_len or feature_type != "unknown" or feature_value is not None:
                    raise ValueError(f"{feature} has more than 1 type.")
                feature_type = ("string", "float", "int64")[i]
                feature_len = each_type_len
                feature_value = each_type_value
            
        feature_values.update({feature: feature_value})
        feature_lens.update({feature: feature_len})
        feature_types.update({feature: feature_type})

    return feature_values, feature_lens, feature_types

def get_first_data_from_tfrecord(tf_record_path):
    """
    解析 并返回 tf_record 的第1条数据
    Returns:
        3个dict, key为特征名, value分别为特征的具体数值(list)，特征的长度，特征的类型("string", "float" 或"int64")
    """
    # 将文件转换为TFRecordDataset. 如果文件有压缩，需要填写compression_type. ".gz"后缀对应"GZIP"
    dataset = tf.data.TFRecordDataset(tf_record_path)
    # 将dataset转为迭代器并取第1个数据
    # 类型是tensorflow.python.framework.ops.EagerTensor
    eager_tensor = dataset.__iter__().__next__()
    example_bytes = eager_tensor.numpy()
    # 使用example解析eager_tensor
    example = tf.train.Example()
    example.ParseFromString(example_bytes)
    # 取出feature的内容
    feature_dict = dict(example.features.feature)

    feature_values, feature_lens, feature_types = get_inofs_from_feature_dict(feature_dict)

    return feature_values, feature_lens, feature_types

if __name__ == "__main__":
    data_path_f = r"/home/zhangliang.thanks/workspace/byted_fedlearner/fedlearner/example/wide_n_deep/data/follower/00.tfrecord"
    values, lens, types = get_first_data_from_tfrecord(data_path_f)

    print('lens: %d', lens)
    print('types: %s', str(types))
    print('values: %s', str(values))

    print(['#' for i in range(80)])

    data_path_l = r"/home/zhangliang.thanks/workspace/byted_fedlearner/fedlearner/example/wide_n_deep/data/leader/00.tfrecord"
    values, lens, types = get_first_data_from_tfrecord(data_path_l)

    print('lens: %d', lens)
    print('types: %s', str(types))
    print('values: %s', str(values))

    