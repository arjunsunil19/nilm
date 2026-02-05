"""Configuration classes for NILM system."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ApplianceConfig:
    """Configuration for a single appliance."""
    name: str
    type: str  # 'binary' or 'multi_state'
    thresholds: List[float]
    state_names: List[str]
    noise_floor: float = 5.0


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_file: str = 'combined_appliances_correct_order.xlsx'
    window_size: int = 99
    midpoint: int = 49
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Feature engineering
    rolling_window: int = 5
    
    # Data preprocessing
    clip_negative: bool = True
    fill_method: str = 'both'  # 'forward', 'backward', or 'both'


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # CNN-BiLSTM-Attention
    cnn_filters: int = 64
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    attention_heads: int = 4
    dropout_rate: float = 0.3
    
    # Training
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    
    # Early stopping
    patience: int = 30
    min_delta: float = 0.0001


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing."""
    median_filter_size: int = 5
    gaussian_sigma: float = 1.0
    apply_physical_bounds: bool = True


@dataclass
class NILMConfig:
    """Main NILM system configuration."""
    
    # Appliances configuration
    appliances: Dict[str, ApplianceConfig] = field(default_factory=lambda: {
        'AC': ApplianceConfig(
            name='Air Conditioner',
            type='binary',
            thresholds=[80],
            state_names=['OFF', 'ON'],
            noise_floor=10.0
        ),
        'Refrigerator': ApplianceConfig(
            name='Refrigerator',
            type='binary',
            thresholds=[20],
            state_names=['OFF', 'ON'],
            noise_floor=5.0
        ),
        'Fan': ApplianceConfig(
            name='Fan',
            type='multi_state',
            thresholds=[10, 25, 45],
            state_names=['OFF', 'Low', 'Medium', 'High'],
            noise_floor=5.0
        ),
        'Washing_Machine': ApplianceConfig(
            name='Washing Machine',
            type='multi_state',
            thresholds=[40, 150, 350],
            state_names=['OFF', 'Wash', 'Rinse', 'Spin'],
            noise_floor=10.0
        ),
        'EV_Charger': ApplianceConfig(
            name='EV Charger',
            type='binary',
            thresholds=[80],
            state_names=['OFF', 'ON'],
            noise_floor=10.0
        )
    })
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    # Output paths
    output_dir: str = 'outputs'
    models_dir: str = 'outputs/models'
    plots_dir: str = 'outputs/plots'
    metrics_dir: str = 'outputs/metrics'
    predictions_dir: str = 'outputs/predictions'
    
    # Model selection
    model_type: str = 'cnn_bilstm_attention'  # Options: cnn_bilstm_attention, transformer, unet, resnet_lstm
    
    # Number of input channels (features)
    n_channels: int = 8
    
    # Random seed for reproducibility
    random_seed: int = 42
