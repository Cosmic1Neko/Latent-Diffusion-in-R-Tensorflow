layer_group_normalization <- function(object, groups = 32, axis = -1, epsilon = 0.001, center = TRUE, 
                                      scale = TRUE, beta_initializer = "zeros", gamma_initializer = "ones", 
                                      beta_regularizer = NULL, gamma_regularizer = NULL, beta_constraint = NULL, 
                                      gamma_constraint = NULL, trainable = TRUE, name = NULL){
  create_layer(keras$layers$GroupNormalization, object, 
               list(groups = as.integer(groups), axis = as.integer(axis), epsilon = epsilon, 
                    center = center, scale = scale, 
                    beta_initializer = beta_initializer, gamma_initializer = gamma_initializer, 
                    beta_regularizer = beta_regularizer, gamma_regularizer = gamma_regularizer, 
                    beta_constraint = beta_constraint, gamma_constraint = gamma_constraint, 
                    trainable = trainable, name = name))
}

variances_embedding <- new_layer_class(
  "VariancesEmbedding",
  # A sinusoidal embedding layer for noise variances embedding
  # used for continuous times diffusion model
  initialize = function(embedding_min_frequency = 1.0,
                        embedding_max_frequency = 10000.0,
                        embedding_dims, ...) {
    super$initialize(...)
    self$embedding_min_frequency <- embedding_min_frequency
    self$embedding_max_frequency <- embedding_max_frequency
    self$embedding_dims <- embedding_dims
    self$frequencies <- tf$exp(
      tf$linspace(
        tf$math$log(self$embedding_min_frequency),
        tf$math$log(self$embedding_max_frequency),
        tf$cast(self$embedding_dims %/% 2, "int32")
      )
    )
    self$angular_speeds <- 2.0 * pi * self$frequencies
  },
  
  call = function(x) {
    x <- tf$cast(x, "float32")
    embeddings <- tf$concat(list(tf$sin(self$angular_speeds * x), tf$cos(self$angular_speeds * x)), axis=-1L)
    return(embeddings)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$embedding_min_frequency <- self$embedding_min_frequency
    config$embedding_max_frequency <- self$embedding_max_frequency
    config$embedding_dims <- self$embedding_dims
    config
  }
)

timestep_embedding <- new_layer_class(
  "TimestepEmbedding",
  # A sinusoidal embedding layer for diffusion timestep embedding
  # used for discrete times diffusion model
  initialize = function(embedding_dims, ...) {
    super$initialize(...)
    self$embedding_dims <- embedding_dims
    self$half_dim <- self$embedding_dims %/% 2
    self$emb <- tf$math$log(10000) / (self$half_dim - 1.0)
    self$emb <- tf$exp(tf$range(self$half_dim, dtype="float32") * (-self$emb))
  },
  
  call = function(x) {
    x <- tf$cast(x, "float32")
    embeddings <- tf$concat(list(tf$sin(x * self$emb), tf$cos(x * self$emb)), axis=-1L)
    return(embeddings)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$embedding_dims <- self$embedding_dims
    config
  }
)

ResidualBlock <- new_layer_class(
  "ResidualBlock",
  initialize = function(width, dropout = 0.0, ...){
    super$initialize(...)
    self$width <- as.integer(width)
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$norm1 <- layer_group_normalization(groups = 16, epsilon = 1e-5, name = "norm1")
    self$conv1 <- layer_conv_2d(filters = self$width, kernel_size = 3, padding = "same", name = "conv1")
    
    self$emb_proj <- layer_dense(units = self$width, name = "emb_proj")
    
    self$norm2 <- layer_group_normalization(groups = 16, epsilon = 1e-5, name = "norm2")
    self$dropout <- layer_dropout(rate = self$dropout, name = "dropout")
    self$conv2 <- layer_conv_2d(filters = self$width, kernel_size = 3, padding = "same", name = "conv2")
    
    if(input_shape[[1]][[4]] == self$width){
      self$residual <- tf$identity
    } else {
      self$residual <- layer_conv_2d(filters = self$width, kernel_size = 1, name = "residual")
    }
  },
  
  call = function(inputs, training = NULL){
    c(x, emb) %<-% inputs
    
    h <- x$shape[[2]]
    w <- x$shape[[3]]
    
    residual <- self$residual(x)
    
    emb_proj <- emb %>% 
      layer_activation(activation = "swish") %>% 
      self$emb_proj() %>% 
      layer_upsampling_2d(size = c(h, w), interpolation = "nearest")
    
    x <- x %>% 
      self$norm1() %>% 
      layer_activation(activation = "swish") %>% 
      self$conv1()
    
    x <- layer_add(x, emb_proj)
    
    x <- x %>% 
      self$norm2() %>% 
      layer_activation(activation = "swish") %>% 
      self$dropout(training = training) %>% 
      self$conv2()
    
    x <- layer_add(x, residual)
    
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$width <- self$width
    config$dropout <- self$dropout
    config
  }
)

BasicTransformerBlock <- new_layer_class(
  "BasicTransformerBlock",
  # It is a standard transformer encoder block.
  # Input shape: (b, t, c)
  # Output shape: (b, t, c)
  initialize = function(num_heads = 8, dropout = 0.0, ...){
    super$initialize(...)
    self$num_heads <- as.integer(num_heads)
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$c <- input_shape[[3]]
    
    self$norm_attn <- layer_layer_normalization(epsilon = 1e-5, name = "norm_attn")
    self$multi_attn <- layer_multi_head_attention(num_heads = self$num_heads, 
                                                  key_dim = self$c %/% self$num_heads,
                                                  name = "multi_attn")
    self$dropout_attn <-layer_dropout(rate = self$dropout, name = "dropout_attn")
    
    self$norm_ffn <- layer_layer_normalization(epsilon = 1e-5, name = "norm_ffn")
    self$ffn1 <- layer_dense(units = self$c, name = "ffn1")
    self$ffn2 <- layer_dense(units = self$c, name = "ffn2")
    self$dropout_ffn <-layer_dropout(rate = self$dropout, name = "dropout_ffn")
  },
  
  call = function(input, training = NULL){
    attn_out <- input %>% 
      self$norm_attn() %>% 
      self$multi_attn(., .) %>% 
      self$dropout_attn(training = training) %>% 
      layer_add(., input)
    
    ffn_out <- attn_out %>% 
      self$norm_ffn() %>%
      self$ffn1() %>% 
      layer_activation(activation = "gelu") %>% 
      self$ffn2() %>% 
      self$dropout_ffn(training = training) %>% 
      layer_add(., attn_out)
    
    return(ffn_out)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$num_heads <- self$num_heads
    config$dropout <- self$dropout
    config
  }
)

SpatialTransformer <- new_layer_class(
  "SpatialTransformer",
  # Transformer block for image-like data.
  # First, project the input (aka embedding) and reshape to (b, t, c).
  # Then apply standard transformer encoder action.
  # Finally, reshape to (b, h, w, c) and project the output.
  # Input shape: (b, h, w, c)
  # Output shape: (b, h, w, c)
  initialize = function(num_blocks = 1, num_heads = 8, dropout = 0.0, ...){
    super$initialize(...)
    self$num_blocks <- as.integer(num_blocks)
    self$num_heads <- as.integer(num_heads)
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$c <- input_shape[[4]]
    
    self$norm_in <- layer_group_normalization(groups = 16, epsilon = 1e-5, name = "norm_in")
    self$proj_in <- layer_conv_2d(filters = self$c, kernel_size = 1, name = "proj_in")
    
    layers <- list()
    for (i in 1:self$num_blocks){
      layers <- c(layers, BasicTransformerBlock(num_heads = self$num_heads, dropout = self$dropout))
    }
    self$transformer_blocks <- keras$Sequential(layers, name = "transformer_blocks")
    
    self$proj_out <- layer_conv_2d(filters = self$c, kernel_size = 1, kernel_initializer = "zeros", name = "proj_out")
  },
  
  call = function(input, training = NULL){
    h <- input$shape[[2]]
    w <- input$shape[[3]]
    
    x <- input %>% 
      self$norm_in() %>% 
      self$proj_in() %>% 
      tf$reshape(shape = c(-1L, h * w, self$c)) # reshape to (b, t, c)
    
    x <- self$transformer_blocks(x)
    
    out <- x %>% 
      tf$reshape(shape = c(-1L, h, w, self$c)) %>%  # reshape to (b, h, w, c)
      self$proj_out() %>%
      layer_add(., input)
    
    return(out)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$num_blocks <- self$num_blocks
    config$num_heads <- self$num_heads
    config$dropout <- self$dropout
    config
  }
)

Downsample <- new_layer_class(
  "Downsample",
  initialize = function(use_conv, ...){
    super$initialize(...)
    self$use_conv <- use_conv
  },
  
  build = function(input_shape){
    self$width <- as.integer(input_shape[[4]])
    if(self$use_conv){
      self$downsample <- layer_conv_2d(filters = self$width, kernel_size = 3, strides = 2, padding = "same")
    } else{
      self$downsample <- layer_average_pooling_2d(pool_size = c(2L,2L))
    }
  },
  
  call = function(input){
    x <- self$downsample(input)
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$use_conv <- self$use_conv
    config
  }
)

Upsample <- new_layer_class(
  "Upsample",
  initialize = function(use_conv, ...){
    super$initialize(...)
    self$use_conv <- use_conv
  },
  
  build = function(input_shape){
    self$width <- as.integer(input_shape[[4]])
    if(self$use_conv){
      self$upsample <- keras_model_sequential(name = "Upsample") %>% 
        layer_upsampling_2d(size = c(2L,2L), interpolation = "nearest") %>% 
        layer_conv_2d(filters = self$width, kernel_size = 3, strides = 1, padding = "same")
    } else{
      self$upsample <- layer_upsampling_2d(size = c(2L,2L), interpolation = "bilinear")
    }
  },
  
  call = function(input){
    x <- self$upsample(input)
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$use_conv <- self$use_conv
    config
  }
)

get_UNET <- function(){
  noisy_images <- layer_input(shape = c(latent_size, latent_size, 4L), name = "noisy_images")
  time_input <- layer_input(shape = c(1L, 1L, 1L), name = "time_input")
  
  if (diffusion_type == "continuous"){
    emb <- time_input %>% 
      variances_embedding(embedding_dims = embedding_dims, dtype = "float32")
  } else if (diffusion_type == "discrete"){
    emb <- time_input %>% 
      timestep_embedding(embedding_dims = embedding_dims, dtype = "float32")
  }
    
  emb <- emb %>% 
    layer_dense(units = embedding_dims * 4) %>% 
    layer_activation(activation = "swish") %>% 
    layer_dense(units = embedding_dims * 4, name = "emb")
  
  skips <- list()
  
  x <- noisy_images %>% 
    layer_conv_2d(widths[1], kernel_size = 3, padding = "same")
  skips <- c(skips,x)
  
  #downsample part
  for(i in 1:length(widths)){
    for(. in 1:num_res_blocks){
      x <- ResidualBlock(list(x ,emb), widths[i], dropout_rate)
      if(has_attention[i]){
        x <- SpatialTransformer(x, num_blocks = num_TransBlock[i], num_heads = 4, dropout = dropout_rate)
      }
      skips <- c(skips,x)
    }
    
    if(i != length(widths)){
      x <- Downsample(x, use_conv = use_conv)
      skips <- c(skips,x)
    }
  }
  
  #middle part
  x <- ResidualBlock(list(x ,emb), widths[length(widths)], dropout_rate)
  x <- SpatialTransformer(x, num_blocks = num_TransBlock[length(num_TransBlock)], num_heads = 4, dropout = dropout_rate)
  x <- ResidualBlock(list(x, emb), widths[length(widths)], dropout_rate)
  
  #upsample part
  index <- length(skips)
  for(i in rev(1:length(widths))){
    for(. in 1:(num_res_blocks + 1)){
      x <- layer_concatenate(list(x,skips[[index]]))
      x <- ResidualBlock(list(x ,emb), widths[i], dropout_rate)
      index <- index - 1
      if(has_attention[i]){
        x <- SpatialTransformer(x, num_blocks = num_TransBlock[i], num_heads = 4, dropout = dropout_rate)
      }
    }
    
    if(i != 1){
      x <- Upsample(x, use_conv = use_conv)
    }
  }
  
  output <- x %>% 
    layer_group_normalization(groups = 16, epsilon = 1e-5) %>% 
    layer_activation(activation = "swish") %>% 
    layer_conv_2d(filter = 4, kernel_size = 3, padding = "same", kernel_initializer = "zeros", dtype = "float32", name = "output")
  
  unet <- keras_model(inputs = list(noisy_images, time_input), outputs = output, name = "residual_unet")
  return(unet)
}