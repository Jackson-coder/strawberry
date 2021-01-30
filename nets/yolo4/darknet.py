import tensorflow as tf

class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * tf.keras.backend.tanh(tf.keras.backend.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def DarknetConv2D_BN_Mish(inputs, num_filters, kernel_size, stride=(1, 1)):
    conv = tf.keras.layers.Conv2D(
        filters=num_filters, kernel_size=kernel_size, strides=stride, padding ='same')(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation(Mish())(conv)

    return conv


def resblock_body(inputs, num_filters, num_blocks, flag=True):
    # 长宽压缩
    # conv = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    # conv = DarknetConv2D_BN_Mish(conv, num_filters, (3, 3), stride=(2, 2))  #convolution部分已经设置参数padding = 'same',图片已经补全像素
    conv = DarknetConv2D_BN_Mish(inputs, num_filters, (3, 3), stride=(2, 2))

    # 建立残差边
    shortconv = DarknetConv2D_BN_Mish(conv,num_filters//2 if flag == True else num_filters,(1,1))

    # 主路
    mainconv = DarknetConv2D_BN_Mish(conv,num_filters//2 if flag == True else num_filters,(1,1))
    for i in range(num_blocks):
        mainconv_v = DarknetConv2D_BN_Mish(mainconv,num_filters//2,(1,1))
        mainconv_v = DarknetConv2D_BN_Mish(mainconv_v,num_filters//2 if flag == True else num_filters,(3,3))
        mainconv = tf.keras.layers.Add()([mainconv,mainconv_v])
    postconv = DarknetConv2D_BN_Mish(mainconv,num_filters//2 if flag == True else num_filters,(1,1))

    #残差边与主路堆叠
    route = tf.keras.layers.Concatenate()([postconv, shortconv])
    
    # 最后对通道数进行整合
    return DarknetConv2D_BN_Mish(route,num_filters,(1,1))



def darknet(inputs):
    x = DarknetConv2D_BN_Mish(inputs, 32, (3, 3))
    print(x.shape)
    x = resblock_body(x, 64, 1, False)
    print(x.shape)
    x = resblock_body(x, 128, 2)
    print(x.shape)
    x = resblock_body(x, 256, 8)
    print(x.shape)
    feat1 = x
    x = resblock_body(x, 512, 8)
    print(x.shape)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    print(x.shape)
    feat3 = x
    return feat1,feat2,feat3


