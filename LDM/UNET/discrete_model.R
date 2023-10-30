# schedule only output alpha_cumprod (alpha_hat), because DDIM don't need beta or alpha to sample
linear_schedule <- function(beta_start = 0.00085, beta_end = 0.0120, timesteps = 1000L){
  # linear beta schedule in "DDPM", sqrt(alpha_cumprod) ~ [0.9995749, 0.03973617]
  scale <- 1000 / timesteps
  betas <- tf$linspace(scale * beta_start, scale * beta_end, timesteps)
  
  alphas <- 1.0 - betas
  alphas_cumprod <- tf$math$cumprod(alphas, axis = 0L)
  return(alphas_cumprod)
}

cosine_schedule <- function(alphas_cumprod_start = 0.999, alphas_cumprod_end = 0.001, timesteps = 1000L){
  # cosine alphas_cumprod(alphas_hat) schedule in "improved DDPM"
  # I set the minimum alphas_cumprod to avoid the extreme value when sampling (1 / sqrt(alphas_cumprod_999) = 1 / 0.03162278)
  # sqrt(alpha_cumprod) ~ [0.99945104, 0.03162471]
  start_angle <- tf$acos(alphas_cumprod_start ^ 0.5)
  end_angle <- tf$acos(alphas_cumprod_end ^ 0.5)
  x <- tf$linspace(1, timesteps, timesteps)
  diffusion_angles <- start_angle + x / timesteps * (end_angle - start_angle)
  alphas_cumprod <- tf$cos(diffusion_angles) ^ 2
  return(alphas_cumprod)
}

DiffusionModel <- new_model_class(
  "DiffusionModel",
  initialize = function(UNET, EMA_UNET, Encoder, Decoder){
    super$initialize()
    self$UNET <- UNET
    self$EMA_UNET <- EMA_UNET
    if(!is.null(self$UNET)){self$EMA_UNET$set_weights(self$UNET$get_weights())}
    self$Encoder <- Encoder
    self$Decoder <- Decoder
    
    # define alphas_cumprod (alphas_hat)
    if (diffusion_schedule == "linear"){
      self$alphas_cumprod <- linear_schedule(beta_start = beta_start, 
                                             beta_end = beta_end, 
                                             timesteps = timesteps)
    } else if (diffusion_schedule == "cosine"){
      self$alphas_cumprod <- cosine_schedule(alphas_cumprod_start = alphas_cumprod_start, 
                                             alphas_cumprod_end = alphas_cumprod_end, 
                                             timesteps = timesteps)
    }
    self$sqrt_alphas_cumprod <- tf$sqrt(self$alphas_cumprod)
    self$sqrt_one_minus_alphas_cumprod <- tf$sqrt(1.0 - self$alphas_cumprod)
  },
  
  compile = function(...){
    super$compile(...)
    self$noise_loss_tracker <- keras$metrics$Mean(name="n_loss")
  },
  
  metrics = mark_active(function() {
    list(self$noise_loss_tracker)
  }),
  
  to_latents = function(images){
    latents <- self$Encoder(images, training = FALSE) * scale_factor
    return(latents)
  },
  
  to_images = function(latents){
    latents <- latents * 1.0 / scale_factor
    images <- self$Decoder(latents, training = FALSE)
    return(images)
  },
  
  extract = function(x, diffusion_times){
    # get the value according to t from the previously defined α_t list.
    return(tf$gather(x, tf$cast(diffusion_times, "int32")))
  },
  
  forward_diffusion = function(latents){
    batch_size <- tf$shape(latents)[1]
    # generate noises and offset noises (https://www.crosslabs.org/blog/diffusion-with-offset-noise)
    noises <- tf$random$normal(shape = c(batch_size, latent_size, latent_size, 4L))
    if (noise_offset != 0){
      noises <- noises + noise_offset * tf$random$normal(shape = c(batch_size, 1L, 1L, 4L))
    }
    
    # generate random diffusion times
    diffusion_times <- tf$random$uniform(shape=c(batch_size, 1L, 1L, 1L), minval=0L, maxval=timesteps, dtype = "int32")
    
    # signal_rates is sqrt(α_t) and noise_rates is sqrt(1 - α_t) in DDIM
    signal_rates <- self$extract(self$sqrt_alphas_cumprod, diffusion_times)
    noise_rates <- self$extract(self$sqrt_one_minus_alphas_cumprod, diffusion_times)
    
    # mix the images with noises accordingly
    noisy_latents <- signal_rates * latents + noise_rates * noises
    
    return(list(noisy_latents, 
                diffusion_times,
                noises))
  },
  
  reverse_process = function(start_x,
                             start_t = timesteps, # used for img2img
                             diffusion_steps,
                             eta = 0.0){
    # reverse diffusion = sampling 
    num_images <- start_x$shape[[1]]
    diffusion_steps <- as.integer(diffusion_steps)
    
    t_seq <- tf$linspace(start_t - 1L, 0L, diffusion_steps) %>% 
      as.integer()
    
    message(paste0("Sampling with steps ", diffusion_steps, ", eta ", eta, " and start_t ", start_t))
    pb <- progress_bar$new(
      format = "[:bar] :current/:total ETA::eta",
      total = diffusion_steps,
      width = 60
    )
    # at the first sampling step, the "x" is pure noise, diffusion_times is 999 and not 1000
    # in img2img, the "x" is noisy images with the "start_t" steps
    x <- start_x 
    for(i in 1:length(t_seq)){
      diffusion_times <- t_seq[i] # get the t
      diffusion_times_prev <- t_seq[i + 1] # get the next t
      if(is.na(diffusion_times_prev)) {last_step = TRUE} else {last_step = FALSE}
      
      diffusion_times <- tf$constant(diffusion_times, shape = c(num_images, 1L, 1L, 1L), dtype = "int32")
      diffusion_times_prev <- tf$constant(diffusion_times_prev, shape = c(num_images, 1L, 1L, 1L), dtype = "int32")
      
      # predict one component of the noisy images with the network
      if (use_ema){
        pred_noises <- self$EMA_UNET(list(x, diffusion_times), training = FALSE)
      } else{
        pred_noises <- self$UNET(list(x, diffusion_times), training = FALSE)
      }
      
      alpha_cumprod <- self$extract(self$alphas_cumprod, diffusion_times)
      # when the last step diffusion_times = 0, diffusion_times_prev = NA, so we set alpha_cumprod_prev = 1.0
      if (!last_step){
        alpha_cumprod_prev <- self$extract(self$alphas_cumprod, diffusion_times_prev)
      } else {
        alpha_cumprod_prev <- tf$constant(1.0, dtype = "float32")
      }
      
      if (all(as.vector(diffusion_times) > 0)){
        noises <- tf$random$normal(shape = x$shape)
      } else {
        noises <- tf$zeros_like(x)
      }
      
      # eta = 0 is DDIM, eta = 1 is fixedsmall variances DDPM
      sigma <- eta * tf$sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod) * (1.0 - alpha_cumprod / alpha_cumprod_prev))
      predict_x0 <- (x - tf$sqrt(1.0 - alpha_cumprod) * pred_noises) / tf$sqrt(alpha_cumprod)
      direction_point <- tf$sqrt(1.0 - alpha_cumprod_prev - tf$square(sigma)) * pred_noises
      random_noise <- sigma * noises
      x <- tf$sqrt(alpha_cumprod_prev) * predict_x0 + direction_point + random_noise # x_{t-1}
      
      pb$tick()
    }
    return(x)
  },
  
  generate = function(num_images,
                      diffusion_steps = 20,
                      eta = 0.0){
    # noise -> latent images -> pixel images
    initial_noise <- tf$random$normal(shape=c(num_images, latent_size, latent_size, 4L))
    latents <- self$reverse_process(start_x = initial_noise,
                                    start_t = timesteps, 
                                    diffusion_steps = diffusion_steps,
                                    eta = eta)
    generated_images <- self$to_images(latents)
    return(generated_images)
  },
  
  img2img = function(images,
                     diffusion_steps = 20,
                     denoising_strength = 0.5,
                     eta = 0.0){
    num_images <- tf$shape(images)[1]
    latents <- self$to_latents(images)
    start_t <- as.integer(denoising_strength * timesteps - 1) %>% # [1,1000] -> [0,999]
      tf$constant(shape = c(num_images, 1L, 1L, 1L))
    
    # forward diffusion
    noises <- tf$random$normal(shape = c(num_images, latent_size, latent_size, 4L))
    signal_rates <- self$extract(self$sqrt_alphas_cumprod, start_t)
    noise_rates <- self$extract(self$sqrt_one_minus_alphas_cumprod, start_t)
    start_x <- signal_rates * latents + noise_rates * noises
    
    # denoising
    latents <- self$reverse_process(start_x = start_x,
                                    start_t = start_t + 1, # [0,999] -> [1,1000]
                                    diffusion_steps = diffusion_steps,
                                    eta = eta)
    
    generated_images <- self$to_images(latents)
    return(generated_images)
  },
  
  train_step = function(images){
    batch_size <- tf$shape(images)[1]
    # images augmentation
    images <- tf$image$random_flip_left_right(images)
    # to latent space
    latents <- self$to_latents(images)
    
    # forward diffusion, add noises to the images
    c(noisy_latents, diffusion_times, noises) %<-% self$forward_diffusion(latents)
    
    # use U-Net to predict the added noises
    with(tf$GradientTape() %as% tape, {
      # train the network to separate noisy images to their components
      pred_noises <- self$UNET(list(noisy_latents, diffusion_times), training=TRUE)
      noise_loss <- self$loss(noises, pred_noises)  # used for training
      if (mixed_precision){
        scaled_loss <- self$optimizer$get_scaled_loss(noise_loss)
      }
    })
    if (mixed_precision){
      scaled_gradients <- tape$gradient(scaled_loss, self$UNET$trainable_weights)
      gradients <- self$optimizer$get_unscaled_gradients(scaled_gradients)
    } else{
      gradients <- tape$gradient(noise_loss, self$UNET$trainable_variables)
    }
    self$optimizer$apply_gradients(
      zip_lists(gradients, self$UNET$trainable_variables)
    )
    
    #ema
    for(w in zip_lists(self$UNET$weights,self$EMA_UNET$weights)){
      w[[2]]$assign(ema * w[[2]] + (1 - ema) * w[[1]])
    }
    
    self$noise_loss_tracker$update_state(noise_loss)
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  },
  
  test_step = function(images){
    batch_size <- tf$shape(images)[1]
    # to latent space
    latents <- self$to_latents(images)
    
    # forward diffusion, add noises to the images
    c(noisy_latents, diffusion_times, noises) %<-% self$forward_diffusion(latents)
    
    # use U-Net to predict the added noises
    pred_noises <- self$UNET(list(noisy_latents, diffusion_times), training=FALSE)
    noise_loss <- self$loss(noises, pred_noises)
    
    self$noise_loss_tracker$update_state(noise_loss)
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  },
  
  plot_images = function(num_images, 
                         diffusion_steps = 20, 
                         eta = 0.0){
    for(i in 1:num_images){
      generated_images <- self$generate(num_images=1L, 
                                        diffusion_steps=diffusion_steps, 
                                        eta=eta)
      generated_images <- tf$image$resize(generated_images, size = c(2L*image_size,2L*image_size))
      generated_images <- as.array(generated_images)
      png(paste0("gen_image",i,".png"), height = 2L*image_size, width = 2L*image_size)
      EBImage::Image(generated_images[1,,,],colormode = "Color") %>% 
        EBImage::transpose() %>% 
        EBImage::normalize() %>% 
        plot()
      dev.off()
    }
  }
)
