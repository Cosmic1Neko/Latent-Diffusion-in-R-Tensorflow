# Rscript train_unet.R
#################### Hyperparameterers ####################  
# data
image_size = 256L
latent_size = image_size %/% 4L
scale_factor = 0.14581
data_dir = "/root/autodl-tmp/512AnimeFaces"
VAE_dir = "/root/autodl-tmp/VAE"
work_dir = "/root/autodl-tmp"

# diffusion algorithmic
diffusion_type = "discrete" # "discrete" or "continuous"
if (diffusion_type == "continuous"){
  min_signal_rate = 0.02
  max_signal_rate = 0.98
} else if (diffusion_type == "discrete"){
  s = 0.008
  timesteps = 1000L
}
noise_offset = 0.0
use_ema = T

# architecture
embedding_dims = 160
widths = c(160, 320, 640) # 64x64, 32x32, 16x16
has_attention = c(FALSE, TRUE, TRUE)
num_TransBlock = c(NA, 1, 3)
num_res_blocks = 2
dropout_rate = 0.0
use_conv = T
   
# optimization
epochs = 50L
batch_size = 20L
learning_rate = 2e-4
weight_decay = 0.01
clipnorm = NULL
ema = 0.999
warmup_steps = 3000
mixed_precision = TRUE










####################  Don't modify the following code ####################  
library(keras)
library(tensorflow)
packageVersion("tensorflow")
library(tfdatasets)
library(abind)
library(EBImage)

source("dataset.R")
source("architecture.R")
source("unet_callback.R")
if (diffusion_type == "continuous"){
  source("continuous_model.R")
} else if (diffusion_type == "discrete"){
  source("discrete_model.R")
}

# load VAE 
VAE <- load_model_tf(VAE_dir)
Encoder <- VAE$get_layer("encoder")
Decoder <- VAE$get_layer("decoder")
Encoder$trainable <- FALSE
Decoder$trainable <- FALSE
rm(VAE);gc()


# Data pipeline
train_dataset <- prepare_dataset(data_dir,
                                 validation_split = 0.01,
                                 subset = "training",
                                 seed = 1919810L)
test_dataset <- prepare_dataset(data_dir,
                                validation_split = 0.01,
                                subset = "validation",
                                seed = 1919810L)

train_dataset %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() %>% 
  as.array() %>% 
  {(. + 1.0) * 127.5} %>% 
  {.[1,,,]} %>% 
  as.raster(max = 255) %>% 
  plot()

test_dataset %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() %>% 
  as.array() %>% 
  {(. + 1.0) * 127.5} %>% 
  {.[1,,,]} %>% 
  as.raster(max = 255) %>% 
  plot()



# mixed precision
# Note: In this script, all the model will output float32, but model still compute with float16
if (mixed_precision){
  tf$keras$mixed_precision$set_global_policy("mixed_float16")
}

lr_schedule <- keras$optimizers$schedules$CosineDecay(initial_learning_rate = learning_rate, 
                                                      warmup_steps = warmup_steps,
                                                      decay_steps = length(train_dataset) * epochs - warmup_steps)
optimizer <- keras$optimizers$AdamW(learning_rate = lr_schedule, weight_decay = weight_decay, clipnorm = clipnorm)
# Exclude Norm layer and bias terms from weight decay.
optimizer$exclude_from_weight_decay(var_names = list("bias", "gamma", "beta"))
if (mixed_precision){
  optimizer <- tf$keras$mixed_precision$LossScaleOptimizer(optimizer)
}

model <- DiffusionModel(UNET = get_UNET(),
                        EMA_UNET = get_UNET(),
                        Encoder = Encoder,
                        Decoder = Decoder)

model %>% compile(
  optimizer = optimizer, 
  loss = keras$losses$mean_squared_error
)


# Train
# Set work dir
setwd(work_dir)

if (!fs::dir_exists("tf-logs")) fs::dir_create("tf-logs")
tensorboard("tf-logs", port = 6007)

checkpoint_filepath = "checkpoint/model_weights"
model_checkpoint_callback <- callback_model_checkpoint(
  filepath = checkpoint_filepath,
  save_best_only = TRUE,
  save_weights_only = TRUE,
  monitor = "n_loss",
  mode = "min",
  save_freq = "epoch",
  verbose = 1)

model %>% fit(train_dataset,
              epoch = epochs,
              validation_data = test_dataset,
              callbacks = list(gan_monitor(save_epochs = 5,plot_diffusion_steps = 20,plot = T),
                               callback_tensorboard(log_dir = "tf-logs",histogram_freq = 1,update_freq=1000L),
                               model_checkpoint_callback)
              )
