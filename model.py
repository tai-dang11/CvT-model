from tensorflow.keras import layers
import tensorflow as tf
from einops.layers.tensorflow import Rearrange
from einops import rearrange
from droppath import DropPath
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, BatchNormalization, AveragePooling2D
from cf import CFGS

class MlpBlock(layers.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features)
        self.fc2 = Dense(out_features)
        self.drop = Dropout(drop)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(layers.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv='same',
                 padding_q='same',
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = layers.Dense(dim_out, use_bias=qkv_bias)
        self.proj_k = layers.Dense(dim_out, use_bias=qkv_bias)
        self.proj_v = layers.Dense(dim_out, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim_out)
        self.proj_drop = layers.Dropout(proj_drop)

        def _build_projection(self,
                            dim_in,
                            dim_out,
                            kernel_size,
                            padding,
                            stride,
                            method):
          if method == 'dw_bn':
              proj = tf.keras.Sequential([
                  tf.keras.layers.Conv2D(
                      filters = dim_in,
                      kernel_size = kernel_size,
                      padding=padding,
                      strides=stride,
                      use_bias=False,
                      groups=dim_in
                  ),
                  BatchNormalization(dim_in),
                  Rearrange('b c h w -> b (h w) c'),
              ])


          elif method == 'avg':
              proj = tf.keras.Sequential([
                  AveragePooling2D(
                      kernel_size=kernel_size,
                      padding=padding,
                      strides=stride,
                      ceil_mode=True
                  ),
                  Rearrange('b c h w -> b (h w) c'),
              ])
          elif method == 'linear':
              proj = None
          else:
              raise ValueError('Unknown method ({})'.format(method))

          return proj




    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = tf.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = tf.concat((cls_token, q), axis=1)
            k = tf.concat((cls_token, k), axis=1)
            v = tf.concat((cls_token, v), axis=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = tf.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = layers.Softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ConvEmbed(layers.Layer):
    """ Image to Conv Embedding
    """

    def __init__(self,
                patch_size=(7,7),
                in_chans=3,
                embed_dim=64,
                strides=4,
                # padding=2,
                padding='same',
                norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=strides,
            padding=padding
        )
        self.norm = norm_layer(epsilon=1e-5)

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.get_shape().as_list()
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
          x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x

class Block(layers.Layer):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 # act_layer=tfa.layers.GELU,
                 norm_layer=LayerNormalization,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else 0.
        self.norm2 = norm_layer(epsilon=1e-5)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = MlpBlock(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            drop=drop
        )

    def forward(self, x, h, w):
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(layers.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=(16,16),
                 patch_stride=16,
                 # patch_padding=0,
                 patch_padding="valid",
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 # act_layer=tfa.layers.GELU,
                 norm_layer=LayerNormalization,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            strides=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = self.add_weight(
                shape=(1, 1, embed_dim),
                initializer=tf.initializers.Zeros())
        else:
            self.cls_token = None

        self.pos_drop = Dropout(rate=drop_rate)
        dpr = [x for x in tf.linspace(0., drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = tf.keras.Sequential([Block(
                                            dim_in=embed_dim,
                                            dim_out=embed_dim,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            drop_path=dpr[j],
                                            # act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            **kwargs)
                                            for j in range(depth)])



    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.get_shape().as_list()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = tf.concat((cls_tokens, x), axis=1)

        x = self.pos_drop(x)

        x = self.blocks(x)

        if self.cls_token is not None:
            cls_tokens, x = tf.split(x, [1, H*W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class ConvolutionalVisionTransformer(layers.Layer):
    def __init__(self,
                 in_chans=3,
                 num_classes=10,
                 norm_layer=LayerNormalization,
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(epsilon=1e-5)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = Dense(num_classes if num_classes > 0. else None, name='head')

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = tf.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = tf.reduce_mean(x, axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def Cvt(model_name='cvt-13-72x72', num_classes=10, CFGS=CFGS):
    CFGS = CFGS[model_name]
    net = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=10,
        norm_layer=LayerNormalization,
        spec = CFGS
    )
    net(tf.keras.Input(shape=(2,CFGS['INPUT_SIZE'][0], CFGS['INPUT_SIZE'][1], 3)))
    return net
