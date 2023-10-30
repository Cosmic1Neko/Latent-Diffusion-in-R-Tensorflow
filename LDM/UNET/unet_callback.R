gan_monitor <- new_callback_class(
  "gan_monitor",
  initialize = function(save_epochs,plot_diffusion_steps,plot) {
    super$initialize()
    self$save_epochs <- save_epochs
    self$plot_diffusion_steps <- plot_diffusion_steps
    self$plot <- plot
    if (!fs::dir_exists("gen_images(UNET)")) fs::dir_create("gen_images(UNET)")
    if (!fs::dir_exists("saved_model(UNET)")) fs::dir_create("saved_model(UNET)")
  },
  on_epoch_end = function(epoch, logs = NULL) {
    #generate images
    if(self$plot){
      generated_images <- model$generate(num_images = 25L, 
                                         diffusion_steps = self$plot_diffusion_steps)
      generated_images <- as.array(generated_images)
      png(paste0("gen_images(UNET)/gen_image(Epoch ",epoch + 1, ").png"))
      par(mfrow=c(5,5))
      for(i in 1:25){
        EBImage::Image(generated_images[i,,,],colormode = "Color") %>% 
          EBImage::transpose() %>% 
          EBImage::normalize() %>% 
          plot(margin = 1)
      }
      dev.off()
    }
    
    #save model
    if((epoch + 1) %% self$save_epochs == 0){
      save_model_tf(model$EMA_UNET,
                    file = paste0("saved_model(UNET)/ema_unet(Epoch ",epoch + 1,")"))
    }
  }
)
