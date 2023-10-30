DiffusionModel <- new_model_class(
  "DiffusionModel",
  initialize = function(UNET, EMA_UNET, Encoder, Decoder){
    super$initialize()
    self$UNET <- UNET
    self$EMA_UNET <- EMA_UNET
    if(!is.null(self$UNET)){self$EMA_UNET$set_weights(self$UNET$get_weights())}
    self$Encoder <- Encoder
    self$Decoder <- Decoder
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
  
  diffusion_schedule = function(diffusion_times){
    # signal_rate = sqrt(α_t)
    # noise_rate = sqrt(1 - α_t)
    # diffusion times -> angles
    start_angle <- tf$acos(alphas_cumprod_start ^ 0.5)
    end_angle <- tf$acos(alphas_cumprod_end ^ 0.5)
    
    diffusion_angles <- start_angle + diffusion_times * (end_angle - start_angle)
    
    # angles -> signal and noise rates
    signal_rates <- tf$cos(diffusion_angles)
    noise_rates <- tf$sin(diffusion_angles)
    return(list("noise_rates" = noise_rates,
                "signal_rates" = signal_rates))
  },
  
  forward_diffusion = function(latents){
    batch_size <- tf$shape(latents)[1]
    # generate noises and offset noises (https://www.crosslabs.org/blog/diffusion-with-offset-noise)
    noises <- tf$random$normal(shape = c(batch_size, latent_size, latent_size, 4L))
    if (noise_offset != 0){
      noises <- noises + noise_offset * tf$random$normal(shape = c(batch_size, 1L, 1L, 4L))
    }
    
    # generate random diffusion times
    diffusion_times <- tf$random$uniform(shape=c(batch_size, 1L, 1L, 1L), minval=0.0, maxval=1.0)
    
    # signal_rates is sqrt(α_t) and noise_rates is sqrt(1 - α_t) in DDIM
    c(noise_rates, signal_rates) %<-% self$diffusion_schedule(diffusion_times)
    
    # mix the images with noises accordingly
    noisy_latents <- signal_rates * latents + noise_rates * noises
    
    return(list(noisy_latents, 
                noise_rates,
                noises))
  },
  
  reverse_process = function(start_x,
                             start_t = 1.0, # used for img2img
                             diffusion_steps,
                             eta = 0.0){
    # reverse diffusion = sampling 
    num_images <- start_x$shape[[1]]
    diffusion_steps <- as.integer(diffusion_steps)
    
    t_seq <- tf$linspace(tf$cast(start_t, "float64"), tf$cast(0.0, "float64"), diffusion_steps) %>% as.numeric()
    
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
      # separate the current noisy image to its components
      diffusion_times <- t_seq[i]
      next_diffusion_times <- t_seq[i + 1]
      if(is.na(next_diffusion_times)) {last_step = TRUE} else {last_step = FALSE}
      
      diffusion_times <- tf$constant(diffusion_times, shape = c(num_images, 1L, 1L, 1L), dtype = "float32")
      next_diffusion_times <- tf$constant(next_diffusion_times, shape = c(num_images, 1L, 1L, 1L), dtype = "float32")
      
      c(noise_rates, signal_rates) %<-% self$diffusion_schedule(diffusion_times)
      if (!last_step){
        c(next_noise_rates, next_signal_rates) %<-% self$diffusion_schedule(next_diffusion_times)
      } else {
        next_noise_rates <- tf$constant(0.0, shape = c(num_images, 1L, 1L, 1L), dtype = "float32")
        next_signal_rates <- tf$constant(1.0, shape = c(num_images, 1L, 1L, 1L), dtype = "float32")
      }
      
      # predict one component of the noisy images with the network
      if (use_ema){
        pred_noises <- self$EMA_UNET(list(x, noise_rates ^ 2), training = FALSE)
      } else{
        pred_noises <- self$UNET(list(x, noise_rates ^ 2), training = FALSE)
      }
      
      if (all(as.vector(diffusion_times) > 0.0)){
        noises <- tf$random$normal(shape = x$shape)
      } else {
        noises <- tf$zeros_like(x)
      }
      
      # eta = 0 is DDIM, eta = 1 is fixedsmall variances DDPM
      sigma <- eta * next_noise_rates / (1.0 - signal_rates ^ 2) * (1.0 - signal_rates ^ 2 / next_signal_rates ^ 2)
      predict_x0 <- (x - noise_rates * pred_noises) / signal_rates
      direction_point <- tf$sqrt(1.0 -  next_signal_rates ^ 2 - tf$square(sigma)) * pred_noises
      random_noise <- sigma * noises
      x <- next_signal_rates * predict_x0 + direction_point + random_noise # x_{t-1}
      
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
                                    start_t = 1.0, 
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
    start_t <- as.numeric(denoising_strength * 1.0) %>% 
      tf$constant(shape = c(num_images, 1L, 1L, 1L))
    
    # forward diffusion
    noises <- tf$random$normal(shape = c(batch_size, latent_size, latent_size, 4L))
    c(noise_rates, signal_rates) %<-% self$diffusion_schedule(start_t)
    start_x <- signal_rates * latents + noise_rates * noises
    
    # denoising
    latents <- self$reverse_process(start_x = start_x,
                                    start_t = start_t,
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
    c(noisy_latents, noise_rates, noises) %<-% self$forward_diffusion(latents)
    
    with(tf$GradientTape() %as% tape, {
      # train the network to separate noisy images to their components
      pred_noises <- self$UNET(list(noisy_latents, noise_rates ^ 2), training=TRUE)
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
    c(noisy_latents, noise_rates, noises) %<-% self$forward_diffusion(latents)
    
    # use the network to separate noisy images to their components
    pred_noises <- self$UNET(list(noisy_latents, noise_rates ^ 2), training=FALSE)
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
