# NeuralFieldTheory
Neural network modelled and studied using field theory

Starting from a string: s="I am a string"

Converting to a list of tokens: i=[4,9,7,5,1]

Then embedding: $x=[ \vec{x}_o,\dots,\vec{x}_T ]$

Convert: $h_t=W_h\cdot x_t$

Get mean: $\bar{z}_t =\sum^{T}_{i} \frac{h_i}{T}$

Obtain a vector and energy: $o_{i,k}=MLP(h_t,\bar{z}_t),\quad E_{i,k}=W_{E}o_{i,k}$

With $k$ corresponding to configuration: $\omega_{i,k}=exp[-\beta(E_{i,k}+b_k)]$

We can compute the partition function: $Z_i=\sum^{K}_{k=1}\omega_{i,k}$

And probability of a configuration: $p_{i,k}=\frac{\omega_{i,k}}{Z_i}$

We can generate outputs based on three different modes:

- Expectation: $y_i=\sum p_{i,k} o_{i,k}$

- Sample: Draw $y_i=o_{i,k}$ with probabilty $p_{i,k}$

- Map:  $y_i=o_{i,k}$ where $k=argmax_k p_{i,k}$

We then apply LayerNorm and finally unembedding to get logits: $logits=W_{ln}LayerNorm(y_i)$