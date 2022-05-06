/// Provides a way to configure [Canvas]
#[derive(Debug, Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct CanvasConfig {
    /// A boolean that will determine where "(0, 0)" - the start of the canvas - is located
    pub upper_left_system: bool,
    /// A boolean that will determine whether to wrap the canvas or not. On by default.
    pub wrapped: bool,
    /// A boolean that will determine whether to possibly create glitch art
    /// It will write ppm files inccorectly
    pub pos_glitch: bool,
    /// Provides a way to animating on canvas [Canvas]
    pub animation_config: AnimationConfig,
}

impl CanvasConfig {
    /// constructor for a new config
    #[must_use]
    pub fn new(upper_left_system: bool, pos_glitch: bool, wrapped: bool) -> Self {
        Self {
            upper_left_system,
            wrapped,
            pos_glitch,
            animation_config: AnimationConfig::default(),
        }
    }

    /// Sets an animation config to the current config
    pub fn set_animation(&mut self, animation_config: AnimationConfig) {
        self.animation_config = animation_config;
    }

    /// Get a reference to the animation config's name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.animation_config.file_prefix.as_ref()
    }

    /// Get a reference to the animation config's file prefix.
    /// # Panics
    /// If animation is not on
    #[must_use]
    pub fn file_prefix(&self) -> &str {
        assert!(self.animation());
        self.animation_config.file_prefix.as_ref()
    }

    /// Get the animation config's anim index.
    #[must_use]
    pub fn anim_index(&self) -> usize {
        self.animation_config.anim_index
    }

    /// Increases the animation config's anim index.
    /// # Panics
    /// If animation is off
    pub fn increase_anim_index(&mut self) {
        assert!(self.animation());
        self.animation_config.anim_index += 1;
    }

    /// Get the animation config's animation.
    #[must_use]
    pub fn animation(&self) -> bool {
        self.animation_config.animation
    }

    /// Set the canvas config's wrapped.
    pub fn set_wrapped(&mut self, wrapped: bool) {
        self.wrapped = wrapped;
    }
}

/// Provides a way to animating on canvas [Canvas]
/// Make sure to access via config.
/// Construct one using `Canvas.set_animation`()
/// Works like this because technically you don't need the other options in config
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct AnimationConfig {
    /// A boolean that will determine whether to create an animation
    animation: bool,
    /// A counter that will be used when saving images for animations
    anim_index: usize,
    /// Prefix!
    file_prefix: String,
}

impl AnimationConfig {
    /// Sets up the configuration for animation
    #[must_use]
    pub fn new(file_prefix: String) -> Self {
        Self {
            animation: true,
            anim_index: Default::default(),
            file_prefix,
        }
    }
}
