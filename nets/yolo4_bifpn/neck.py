import tensorflow as tf


def DarknetConv2D_BN_Leaky(inputs, num_filters, kernel_size, stride=(1, 1)):
    conv = tf.keras.layers.Conv2D(
        filters=num_filters, kernel_size=kernel_size, strides=stride, padding='same')(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation(
        tf.keras.layers.LeakyReLU(alpha=0.1))(conv)

    return conv



def spp(inputs):
    conv = DarknetConv2D_BN_Leaky(inputs, 512, (1, 1))
    conv = DarknetConv2D_BN_Leaky(conv, 1024, (3, 3))
    conv = DarknetConv2D_BN_Leaky(conv, 512, (1, 1))

    P1 = tf.keras.layers.MaxPool2D(pool_size=(
        13, 13), strides=(1, 1), padding='same')(conv)
    print(P1)
    P2 = tf.keras.layers.MaxPool2D(pool_size=(
        9, 9), strides=(1, 1), padding='same')(conv)
    print(P2)
    P3 = tf.keras.layers.MaxPool2D(pool_size=(
        5, 5), strides=(1, 1), padding='same')(conv)
    print(P3)
    P = tf.keras.layers.Concatenate(axis=3)([P1, P2, P3, conv])
    print(P)

    P = DarknetConv2D_BN_Leaky(P, 512, (1, 1))
    P = DarknetConv2D_BN_Leaky(P, 1024, (3, 3))
    P = DarknetConv2D_BN_Leaky(P, 512, (1, 1))

    return P


def make_five_convs(x, num_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(x, num_filters, (1, 1))
    x = DarknetConv2D_BN_Leaky(x, num_filters*2, (3, 3))
    x = DarknetConv2D_BN_Leaky(x, num_filters, (1, 1))
    x = DarknetConv2D_BN_Leaky(x, num_filters*2, (3, 3))
    x = DarknetConv2D_BN_Leaky(x, num_filters, (1, 1))
    return x

# input
# 52 52 256
# 26 26 512
# 13 13 1024


def bifpn(input1, input2, input3):
    # upsample
    # 13,13,512 -> 13,13,256 -> 26,26,256
    P1_U = DarknetConv2D_BN_Leaky(input3, 256, (1, 1))
    P1_U = tf.keras.layers.UpSampling2D()(P1_U)
    # 26,26,512 -> 26,26,256 
    P2 = DarknetConv2D_BN_Leaky(input2, 256, (1, 1))
    # 26,26,256 + 26,26,256 -> 26,26,512
    P2_td = tf.keras.layers.Concatenate(axis=3)([P1_U, P2])
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P2_td = make_five_convs(P2_td,256)
    # 52,52,256 -> 52,52,128
    P3 = DarknetConv2D_BN_Leaky(input1, 128, (1, 1))
    # 26,26,256 -> 26,26,128
    P2_td_U = DarknetConv2D_BN_Leaky(P2_td, 128, (1, 1))
    # 26,26,256 -> 52,52,128
    P2_td_U = tf.keras.layers.UpSampling2D()(P2_td_U)
    # 52,52,128 + 52,52,128 -> 52,52,256
    P3_out = tf.keras.layers.Concatenate(axis=3)([P2_td_U,P3])
    # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    P3_out = make_five_convs(P3_out,128)
    

    # downsample
    # 52,52,128 -> 26,26,256
    P3_out_D = DarknetConv2D_BN_Leaky(P3_out, 256, kernel_size=(3, 3), stride=(2, 2))
    # 26,26,256 + 26,26,256 + 26,26,256 -> 26,26,768
    P2_out = tf.keras.layers.Concatenate(axis=3)([P2, P3_out_D,P2_td])
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P2_out = make_five_convs(P2_out, 256)  
    # 26,26,256 -> 13,13,512
    P2_out_D = DarknetConv2D_BN_Leaky(P2_out, 512, kernel_size=(3, 3), stride=(2, 2))
    # 13,13,512 + 13,13,512 -> 13,13,1024
    P1_out = tf.keras.layers.Concatenate(axis=3)([input3, P2_out_D])
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P1_out = make_five_convs(P1_out, 512)  

    # 52,52,128, 26,26,256, 13,13,512
    return P3_out,P2_out,P1_out

def yolo_head(P6, P7, P8, num_anchors, num_classes):
    batch_size = num_anchors*(num_classes + 5)

    P6_output = DarknetConv2D_BN_Leaky(P6, 256, kernel_size=(3, 3))
    P6_output = tf.keras.layers.Conv2D(
        filters=batch_size, kernel_size=(1, 1))(P6_output)

    P7_output = DarknetConv2D_BN_Leaky(P7, 512, kernel_size=(3, 3))
    P7_output = tf.keras.layers.Conv2D(
        filters=batch_size, kernel_size=(1, 1))(P7_output)

    P8_output = DarknetConv2D_BN_Leaky(P8, 1024, kernel_size=(3, 3))
    P8_output = tf.keras.layers.Conv2D(
        filters=batch_size, kernel_size=(1, 1))(P8_output)

    return P6_output, P7_output, P8_output


def neck_and_head(feat1, feat2, feat3, num_anchors, num_classes):
    feat3 = spp(feat3)
    P6, P7, P8 = bifpn(feat1, feat2, feat3)
    return yolo_head(P6, P7, P8, num_anchors, num_classes)