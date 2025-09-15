import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import make_proba_distribution

class GasEnvFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super(GasEnvFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Gas indicators input dimension: 6 features (CH4, N2O, NO3, N_runoff, NH3, SOC)

        self.fc_gases = nn.Linear(6, features_dim)
        self.layernorm_gases = nn.LayerNorm(features_dim)
        
        # Tech_selected input dimension: numTech + 1 (assuming numTech is known from observation_space)
        num_tech = observation_space['Tech_selected'].shape[1] - 1  # Extract numTech from shape
        self.fc_tech = nn.Linear(num_tech + 1, features_dim)
        self.layernorm_tech = nn.LayerNorm(features_dim)
        
        # Attention and Transformer layers
        self.attention_layer = nn.MultiheadAttention(embed_dim=features_dim, num_heads=4, batch_first=True)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=4)
        
        # Final fully connected layer for output features
        self.fc_final = nn.Linear(features_dim, features_dim)
        
        # Dummy input to initialize (optional, ensures forward pass works during initialization)
        dummy_input = {
            "CH4": th.zeros((1,) + observation_space['CH4'].shape),
            "N2O": th.zeros((1,) + observation_space['N2O'].shape),
            "NO3": th.zeros((1,) + observation_space['NO3'].shape),
            "N_runoff": th.zeros((1,) + observation_space['N_runoff'].shape),
            "NH3": th.zeros((1,) + observation_space['NH3'].shape),
            "SOC": th.zeros((1,) + observation_space['SOC'].shape),
            "Tech_selected": th.zeros((1,) + observation_space['Tech_selected'].shape),
        }
        self.forward(dummy_input)

    def forward(self, observations):
        device = next(self.parameters()).device  # 获取当前模型参数所在 device
        # Extract gas indicators from the observation dictionary
        # 把 observation 中的每个 tensor 显式转到 device 上
        ch4 = observations["CH4"].to(device)
        n2o = observations["N2O"].to(device)
        no3 = observations["NO3"].to(device)
        n_runoff = observations["N_runoff"].to(device)
        nh3 = observations["NH3"].to(device)
        soc = observations["SOC"].to(device)
        tech_selected = observations["Tech_selected"].to(device)
        
        # Concatenate gas indicators along the feature dimension
        gases = th.cat([ch4, n2o, no3, n_runoff, nh3, soc], dim=2)  # (batch_size, numCounty, 6)
        
        # Extract technology selection
        tech_selected = observations["Tech_selected"]  # (batch_size, numCounty, numTech + 1)
        
        # normalize gas indicators
        gases_min = gases.min()
        if gases_min < 0:
            gases = gases - gases_min + 1e-6  # Shift to make all values positive
        gases = th.log1p(gases)  # log(1 + x) to avoid log(0)
        
        mean = gases.mean(dim=[0, 1], keepdim=True)
        std = gases.std(dim=[0, 1], keepdim=True) + 1e-5  # Avoid division by zero
        gases = (gases - mean) / std
        
        gases_features = th.relu(self.fc_gases(gases))  # (batch_size, numCounty, features_dim)
        gases_features = self.layernorm_gases(gases_features)
        
        # Process technology selection
        tech_features = th.relu(self.fc_tech(tech_selected))  # (batch_size, numCounty, features_dim)
        tech_features = self.layernorm_tech(tech_features)
        
        # Combine gas and tech features
        combined_features = gases_features + tech_features  # Element-wise addition
        
        # Apply Transformer layer
        transformer_output = self.transformer_layer(combined_features)  # (batch_size, numCounty, features_dim)
        
        # Apply attention mechanism to fuse features across counties
        attn_output, _ = self.attention_layer(transformer_output, transformer_output, transformer_output)
        
        # Pool features across counties (mean pooling)
        pooled_features = th.mean(attn_output, dim=1)  # (batch_size, features_dim)
        
        # Final output
        final_features = th.relu(self.fc_final(pooled_features))  # (batch_size, features_dim)
        
        return final_features

class GasEnvPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(GasEnvPolicy, self).__init__(*args, **kwargs,
                                           features_extractor_class=GasEnvFeatureExtractor,
                                           features_extractor_kwargs=dict(features_dim=128))
        
        # Set action distribution for MultiDiscrete action space
        self.action_dist = make_proba_distribution(self.action_space)