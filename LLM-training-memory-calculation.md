Use the following prior knowledge to write a javascript/html based program to interactively calculate a transformer decoder model's parameter number and its memory consumption
The user inputs are   
1) vocab size - text input
2) hidden size - text input
3) number of attention head (head dimention can be infered by hidden_size/num_of_heads) - input list (8,16,24 ... 128)
4) q/k/v bias (check box)
4) norm type - input list (layernor/rmsnorm)
5) dense intermediate size - text input 
6) router expert intermediate size - text input
7) shared expert intermediate size - text input
8) number of router experts - text input
9) number of shared experts - text input
Please output number of parameters as well as memory consumption of parameters. Please also show how the calculation is done (leveraging the detailed formula as below) to user

## LLM (transformer decoder) training memory consumption
### Model parameter memory
- Word embedding  
$$
VocabSize * HiddenSize * NumberBytesPerDataType
$$

#### Per layer parameters  
- Pre-attention norm   
    - layernorm
    $$
    HiddenSize*2*NumberBytesPerDataType
    $$
    Multiply by 2 - 1 for layernorm weights, 1 for layernorm bias
    - rmsnorm 
    $$	    
    hidden_size*NumberBytesPerDataType
    $$
    rmsnorm doesn't have bias

- qkv projection   

    For Q,
    $$
    HiddenSize * HeadDim * NumberOfAttentionHead + HeadDim * NumberOfAttentionHead
    $$
    For K/V,
    $$
    HiddenSize * HeadDim * NumberOfQueryGroups + HeadDim * NumberOfAttentionHead
    $$
The second term is only applicable if the model uses q/k/v bias. So for Q+K+V projection, total memory consumption is
$$
HiddenSize * HeadDim * NumberOfAttentionHead + HeadDim * NumberOfAttentionHead + \\
    (HiddenSize * HeadDim * NumberOfQueryGroups + HeadDim * NumberOfAttentionHead) * 2
$$

- attention output   
$$
    HiddenSize * HiddenSize * NumberBytesPerDataType
$$

- post attention norm   
    same with pre attention norm
- MLP (assume swiGlu)
	- dense layer:   
    $$
		HiddenSize * DenseIntermediateSize * 3 * NumberBytesPerDataType
    $$
	- moe layer:   
	    - router expert weights:   
            $$
                HiddenSize * MoeRouterExpertIntermediateSize * 3 * NumberOfRouterExperts * NumberBytesPerDataType
            $$
		- share expert weights (if applicable):   
            $$
                HiddenSize * MoeSharedExpertIntermediateSize * 3 * NumberOfSharedExperts * NumberBytesPerDataType
            $$
        - shared expert gate (if applicable)
            $$
                HiddenSize * NumberBytesPerDataType
            $$
        - total moe layer
            $$
                HiddenSize * MoeRouterExpertIntermediateSize * 3 * NumberOfRouterExperts * NumberBytesPerDataType + \\ HiddenSize * MoeSharedExpertIntermediateSize * 3 * NumberOfSharedExperts * NumberBytesPerDataType \\ + HiddenSize * NumberBytesPerDataType
	- moe router:   
        $$
	        HiddenSize * NumberOfExpert * NumberBytesPerRouterDataType
        $$
- output layernorm   
    same as pre attention norm

output layer:   
    same as word embedding layer

#### Total parameter memory   
Sum of parameter memory above, per-layer parameters needs to multiply number of layers of the model

$$
Total Parameter Memory = Word Embedding + \\
(Pre-attention norm  + qkv projection + attention output + post attention norm + MLP)  \\
* layer number + outputlayernorm + output layer
$$

- For example,   
```
vocab_size = 152064   
hidden_size = 2048   
number_attention_head = 16
number_query_groups = 16
head_dim=hidden_size/number_attention_head=128
number_router_experts = 60
number_shared_experts = 1
router_expert_intermediate_size = 1408
shared_expert_intermediate_size = 5632
norm_type=rmsnorm
data_type=bf16
router_data_type=fp32
number_of_layer=24
All the layers are expert layers

```
Embedding:
```
vocab_size*hidden_size*number_bytpes_data_type=152064*2048*2=622,854,144
```
Per-layer parameters:   
pre-attn norm:
```
hidden_size*number_bytes_data_type=2048*2=4096
```
q projection:   
```
hidden_size*number_of_attention_head*hidden_dim*number_bytes_data_type=(2048*16*128+16*128)*2=8,392,704
```
k/v projection:
```
hidden_size*number_of_query_groups*hidden_dim*number_bytes_data_type=(2048*16*128*2+16*128)*2=8,392,704
```
attention output:
```
hidden_size*hidden_size*number_bytes_data_type=2048*2048*2=8,388,608
```
post-attn norm:
```
hidden_size*number_bytes_data_type=2048*2=4096
```
MLP:  
- router expert weights:   
```
hidden_size*router_expert_intermediate_size*3*number_of_router_experts*number_bytes_data_type=2048*1408*3*60*2=1,038,090,240
```

- shared expert weights:   
```
hidden_size*shared_expert_intermediate_size*3*number_of_shared_experts*number_bytes_data_type=2048*5632*3*2=69,206,016
```
- shared expert gate:
```
hidden_size*number_of_shared_experts*number_bytes_data_type = 2048*2 = 4096
```
- moe router:   
```
hidden_size*number_of_router_experts*number_bytes_router_data_type=2048*60*2=245,760
```
- taotal MLP:
```
1,038,090,240+69,206,016+245,760+4096=1,107,546,112
```
output norm:
```
hidden_size*number_bytes_data_type=2048*2=4096
```
output layer:
```
hidden_size*vocab_size*number_bytes_data_type=152064*2048*2=622,854,144
```
- Total parameter count:   
Leverage memory consumption formula but without multiplying the byte-of-data-type term
```
Total Paramter Count = Word Embedding + (Pre-attention norm  + qkv projection + attention output + post attention norm + MLP)  * layer number + outputlayernorm + output layer = 152064*2048 + (2048 + (2048*16*128+16*128)+(2048*16*128+16*128)*2 + 2048*2048+ 2048 + (2048*1408*3*60+2048*5632*3+2048*60 + 2048))*24+2048+152064*2048 = 14,316,308,480

```
- Total parameter memory:
```
Total Parameter Memory = Word Embedding + (Pre-attention norm  + qkv projection + attention output + post attention norm + MLP)  * layer number + outputlayernorm + output layer = 622,854,144+(4096+(8,392,704+8,392,704*2)+8,388,608+4096+1,107,546,112)*24+4096+622,854,144=28,632,616,960
```


14,316,308,480


Now, let's add more functionality to this program. First let's add tensor model parallel (tp), pipeline model parallel (pp), data model parallel (dp), expert model parallel (ep), sequence parallel (sp) and distributed optimizer (do) to the horizon.

After applying parallelism, the model parameter memory required for a specific rank should be calculated by this
$$
Model Parameter Memory Per Ra


add gradients and optimizer states required for training the model. Specifically the following formulas rule the memory consumption of both.

#### Gradient memory
$$
TotalNumberOfParameters*GradientDataTypeBytes/(TensorModelParallelSize*PipelineModelParallelSize)
