# Rscript scale_factor.R
library(keras)
library(tensorflow)
packageVersion("tensorflow")
library(tfdatasets)
library(abind)
library(EBImage)

image_size = 64L
batch_size = 8L

# Set work dir
#setwd("/root/autodl-tmp")

VAE <- load_model_tf("/mnt/c/Users/22877/Desktop/Anime Face/VAE")
Encoder <- VAE$get_layer("encoder")
Decoder <- VAE$get_layer("decoder")
Encoder$trainable <- FALSE
Decoder$trainable <- FALSE
rm(VAE);gc()

load_image <- function(img_path) {
  img <-  img_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$resize(size = c(image_size,image_size),antialias = T) %>% 
    layer_rescaling(scale = 1.0/127.5, offset = -1.0, dtype = "float32")
  return(tf$clip_by_value(img,-1.0,1.0))
}
  
prepare_dataset <- function(file_path,validation_split,seed,subset){
  image_count <- list.files(file_path) %>% length()
  list_ds <- tf$data$Dataset$list_files(paste0(file_path,"/*"),shuffle = T, seed = seed)
  val_size <- as.integer(image_count * validation_split)
  if(subset == "training"){
    return(list_ds %>% dataset_skip(val_size) %>% 
           dataset_map(load_image,num_parallel_calls=tf$data$AUTOTUNE) %>% 
           dataset_shuffle(buffer_size = 1024L) %>% 
           dataset_batch(batch_size) %>% 
           dataset_prefetch(buffer_size = tf$data$AUTOTUNE))
  } else if(subset == "validation"){
    return(list_ds %>% dataset_take(val_size) %>% 
           dataset_map(load_image,num_parallel_calls=tf$data$AUTOTUNE) %>%
           dataset_shuffle(buffer_size = 1024L) %>% 
           dataset_batch(batch_size) %>% 
           dataset_prefetch(buffer_size = tf$data$AUTOTUNE))
  }
}

train_dataset <- prepare_dataset("/mnt/c/Users/22877/Desktop/Anime Face/faces",
                                 validation_split = 0.01,
                                 subset = "training",
                                 seed = 1919810L)
test_dataset <- prepare_dataset("/mnt/c/Users/22877/Desktop/Anime Face/faces",
                                 validation_split = 0.01,
                                 subset = "validation",
                                 seed = 1919810L)

compute_scale_factor <- new_model_class(
  "compute_scale_factor",
  initialize = function(encoder){
    super$initialize()
    self$encoder <- encoder
  },
  
  compile = function(){
    super$compile()
    self$var_tracker <- keras$metrics$Mean(name="var")
  },
  
  metrics = mark_active(function() {
    list(self$var_tracker)
  }),
  
  train_step = function(images){
    latent_images <- self$encoder(images, training = FALSE)
    mean <- tf$reduce_mean(latent_images)
    var <- tf$reduce_mean((latent_images - mean) ^ 2)
    
    self$var_tracker$update_state(var)
    
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  }
)

model <- compute_scale_factor(encoder = Encoder)
model %>% compile()
model %>% fit(train_dataset, epoch = 1)


std <- model$var_tracker$result() %>% 
  tf$sqrt()
scale_factor = 1.0 / std
print(scale_factor)