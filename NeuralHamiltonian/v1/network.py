import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For attention scaling

# Helper Attention Module
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model) # For decoder query
        self.W_k = nn.Linear(d_model, d_model) # For encoder key
        self.W_v = nn.Linear(d_model, d_model) # For encoder value
        self.W_o = nn.Linear(d_model, d_model) # Output projection

    def forward(self, query_dec, key_enc, value_enc, enc_padding_mask=None):
        # query_dec: (batch, seq_len_dec, d_model) - current decoder q state for one token
        # key_enc:   (batch, seq_len_enc, d_model) - all encoder q states
        # value_enc: (batch, seq_len_enc, d_model) - all encoder q states (or p states, or concat)
        # enc_padding_mask: (batch, seq_len_enc) - True for non-padded encoder tokens

        batch_size = query_dec.size(0)

        # Project and reshape for multi-head
        # Q: (batch, seq_len_dec, num_heads, d_k) -> (batch, num_heads, seq_len_dec, d_k)
        q = self.W_q(query_dec).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # K: (batch, seq_len_enc, num_heads, d_k) -> (batch, num_heads, seq_len_enc, d_k)
        k = self.W_k(key_enc).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # V: (batch, seq_len_enc, num_heads, d_k) -> (batch, num_heads, seq_len_enc, d_k)
        v = self.W_v(value_enc).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        # scores: (batch, num_heads, seq_len_dec, seq_len_enc)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if enc_padding_mask is not None:
            # enc_padding_mask: (batch, seq_len_enc) -> (batch, 1, 1, seq_len_enc) for broadcasting
            mask = enc_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9) # Fill padded positions with a large negative value

        attn_weights = F.softmax(scores, dim=-1) # (batch, num_heads, seq_len_dec, seq_len_enc)
        
        # context: (batch, num_heads, seq_len_dec, d_k)
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        # context: (batch, seq_len_dec, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context) # (batch, seq_len_dec, d_model)
        return output, attn_weights


class HamiltonianEncoder(nn.Module):
    def __init__(self, vocab_size, d_embedding, sequence_length, d_hidden_potential,
                 num_hamiltonian_steps=5, h_step_integrator=0.1, delta_t_momentum=1.0,
                 force_clip_value=None, pad_idx=0):
        super().__init__()
        self.d_embedding = d_embedding
        self.sequence_length = sequence_length
        self.num_hamiltonian_steps = num_hamiltonian_steps
        self.h_step_integrator = h_step_integrator
        self.register_buffer('delta_t_momentum', torch.tensor(delta_t_momentum, dtype=torch.float32))
        self.force_clip_value = force_clip_value
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=self.pad_idx)
        self.log_masses = nn.Parameter(torch.zeros(sequence_length)) # Init masses to 1

        # Potential MLP - can be simpler for encoder, or could also have self-attention
        self.potential_mlp = nn.Sequential(
            nn.Linear(d_embedding, d_hidden_potential),
            nn.Tanh(),
            nn.Linear(d_hidden_potential, 1)
        )
        # Optional: Add positional encodings here if needed
        # self.pos_encoder = PositionalEncoding(d_embedding, dropout_p)


    def get_masses(self):
        return torch.exp(self.log_masses).unsqueeze(0).unsqueeze(-1)

    def _calculate_potential_energy_and_forces(self, q_state, padding_mask):
        q_state_for_V = q_state.detach().requires_grad_(True)
        with torch.enable_grad():
            V_per_token = self.potential_mlp(q_state_for_V)
            if padding_mask is not None:
                V_per_token = V_per_token * padding_mask.unsqueeze(-1).float()
            V_total_scalar = V_per_token.sum()
            forces = -torch.autograd.grad(V_total_scalar, q_state_for_V, create_graph=True, retain_graph=True)[0]
        
        if padding_mask is not None:
            forces = forces * padding_mask.unsqueeze(-1).float()
        if self.force_clip_value is not None:
            forces = torch.clamp(forces, -self.force_clip_value, self.force_clip_value)
        return V_per_token.sum(dim=(1,2)).detach(), forces

    def _symplectic_step(self, q_current, p_current, padding_mask):
        masses = self.get_masses().to(q_current.device)
        _, forces_at_q_current = self._calculate_potential_energy_and_forces(q_current, padding_mask)
        p_half = p_current + (self.h_step_integrator / 2.0) * forces_at_q_current
        q_delta = self.h_step_integrator * (p_half / masses)
        if padding_mask is not None: q_delta = q_delta * padding_mask.unsqueeze(-1).float()
        q_next = q_current + q_delta
        _, forces_at_q_next = self._calculate_potential_energy_and_forces(q_next, padding_mask)
        p_next = p_half + (self.h_step_integrator / 2.0) * forces_at_q_next
        if padding_mask is not None: p_next = p_next * padding_mask.unsqueeze(-1).float()
        return q_next, p_next

    def forward(self, enc_input_ids, enc_padding_mask=None):
        if enc_input_ids.shape[1] != self.sequence_length:
             raise ValueError(f"Encoder input length {enc_input_ids.shape[1]} != model sequence_length {self.sequence_length}")

        q_initial = self.embedding(enc_input_ids)
        # q_initial = self.pos_encoder(q_initial) # If using positional encoding

        if enc_padding_mask is None:
            enc_padding_mask = (enc_input_ids != self.pad_idx)

        padded_q = F.pad(q_initial, (0, 0, 1, 0), mode='replicate')
        velocities = (q_initial - padded_q[:, :-1, :]) / self.delta_t_momentum.to(q_initial.device)
        p_initial = self.get_masses().to(q_initial.device) * velocities
        p_initial = p_initial * enc_padding_mask.unsqueeze(-1).float()

        q_current, p_current = q_initial, p_initial
        # Optional: store trajectories if needed for deeper attention
        # all_q_steps = [q_current] 
        # all_p_steps = [p_current]

        for _ in range(self.num_hamiltonian_steps):
            q_current, p_current = self._symplectic_step(q_current, p_current, enc_padding_mask)
            # all_q_steps.append(q_current)
            # all_p_steps.append(p_current)
        
        # Return final q and p states (or all states if needed for complex attention)
        return q_current, p_current # (batch, seq_len_enc, d_embed)


class HamiltonianDecoder(nn.Module):
    def __init__(self, vocab_size, d_embedding, sequence_length, d_hidden_potential,
                 num_hamiltonian_steps=5, h_step_integrator=0.1, delta_t_momentum=1.0, # Not used for p_initial in dec
                 force_clip_value=None, pad_idx=0, d_encoder_output=None, num_attn_heads=4):
        super().__init__()
        self.d_embedding = d_embedding
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length # Decoder sequence length
        self.num_hamiltonian_steps = num_hamiltonian_steps
        self.h_step_integrator = h_step_integrator
        self.force_clip_value = force_clip_value
        self.pad_idx = pad_idx
        self.d_encoder_output = d_encoder_output if d_encoder_output is not None else d_embedding

        self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=self.pad_idx)
        self.log_masses = nn.Parameter(torch.zeros(sequence_length)) # Init masses to 1

        self.cross_attention = CrossAttention(d_model=d_embedding, num_heads=num_attn_heads)
        
        # Potential MLP input: current q_token (d_embedding) + attended context (d_encoder_output)
        self.potential_mlp = nn.Sequential(
            nn.Linear(d_embedding + self.d_encoder_output, d_hidden_potential),
            nn.Tanh(),
            nn.Linear(d_hidden_potential, 1)
        )
        self.output_layer_logits = nn.Linear(d_embedding, vocab_size)
        # self.pos_encoder = PositionalEncoding(d_embedding, dropout_p)

    def get_masses(self):
        return torch.exp(self.log_masses).unsqueeze(0).unsqueeze(-1)

    def _calculate_potential_energy_and_forces_dec(self, q_token_dec, encoder_q_outputs, enc_padding_mask):
        # q_token_dec: (batch, 1, d_embedding) - current decoder q for a single time step
        # encoder_q_outputs: (batch, seq_len_enc, d_encoder_output)
        
        # 1. Cross-Attention
        # query is q_token_dec, key/value are encoder_q_outputs
        attended_encoder_context, _ = self.cross_attention(q_token_dec, encoder_q_outputs, encoder_q_outputs, enc_padding_mask)
        # attended_encoder_context: (batch, 1, d_encoder_output)

        # 2. Prepare input for potential MLP
        # Ensure q_token_dec is set up for grad for this local calculation
        q_token_dec_for_V = q_token_dec.detach().requires_grad_(True)
        
        # Concatenate q_token with its attended context
        # We detach attended_encoder_context as we don't want to backprop through encoder params *during force calculation for decoder*
        # The gradients to encoder params will flow via the loss on the final decoder output, through the attention weights.
        potential_mlp_input = torch.cat([q_token_dec_for_V, attended_encoder_context.detach()], dim=-1) # (batch, 1, d_embed + d_enc_out)

        with torch.enable_grad():
            # V_per_token here is actually V for this single q_token_dec
            V_this_token = self.potential_mlp(potential_mlp_input) # (batch, 1, 1)
            V_total_scalar = V_this_token.sum() 
            # Differentiate V_this_token w.r.t q_token_dec_for_V only
            # The gradient dV/d(attended_context) is not needed for symplectic update of q_token_dec
            forces = -torch.autograd.grad(V_total_scalar, q_token_dec_for_V, create_graph=True, retain_graph=True)[0]
            # forces shape: (batch, 1, d_embedding)

        if self.force_clip_value is not None:
            forces = torch.clamp(forces, -self.force_clip_value, self.force_clip_value)
        
        return V_this_token.sum().detach(), forces # sum over batch and singleton dims

    def _symplectic_step_dec(self, q_current_token, p_current_token, encoder_q_outputs, enc_padding_mask):
        # Operates on a single token's q and p state at a time, but uses full encoder_q_outputs for context
        # q_current_token, p_current_token: (batch, 1, d_embedding)
        masses_for_this_token = self.get_masses()[:, 0:1, :] # Assuming masses are for decoder positions, take first for example if not step-dependent
                                                        # This needs refinement if masses are per decoder step

        _, forces_at_q_current = self._calculate_potential_energy_and_forces_dec(q_current_token, encoder_q_outputs, enc_padding_mask)
        p_half = p_current_token + (self.h_step_integrator / 2.0) * forces_at_q_current
        
        q_delta = self.h_step_integrator * (p_half / masses_for_this_token) # Problem: masses are seq_len long
        q_next_token = q_current_token + q_delta
        
        _, forces_at_q_next = self._calculate_potential_energy_and_forces_dec(q_next_token, encoder_q_outputs, enc_padding_mask)
        p_next_token = p_half + (self.h_step_integrator / 2.0) * forces_at_q_next
        
        return q_next_token, p_next_token

    def forward(self, dec_input_ids, dec_padding_mask, encoder_q_outputs, enc_padding_mask):
        # dec_input_ids: (batch, seq_len_dec) - Target sequence (shifted right)
        # dec_padding_mask: (batch, seq_len_dec)
        # encoder_q_outputs: (batch, seq_len_enc, d_encoder_output)
        # enc_padding_mask: (batch, seq_len_enc)

        batch_size, seq_len_dec = dec_input_ids.shape
        
        q_embedded_dec = self.embedding(dec_input_ids) # (batch, seq_len_dec, d_embed)
        # q_embedded_dec = self.pos_encoder(q_embedded_dec)

        # Initialize p_initial_dec to zeros
        p_initial_dec = torch.zeros_like(q_embedded_dec) # (batch, seq_len_dec, d_embed)

        all_output_logits = []
        
        # Current simplification: Evolve each decoder token's (q,p) state somewhat independently
        # using its specific attention context from the encoder for its V.
        # A more complex version would have the (q_all_dec, p_all_dec) state evolve together.
        # This loop processes token by token for Hamiltonian evolution, which is not ideal for batching Hamiltonian steps efficiently.
        # A full Hamiltonian evolution of the *entire decoder sequence* conditioned on encoder output is more aligned,
        # but the current _calculate_potential_energy_and_forces_dec is designed for one decoder q_token.
        # Let's adjust to evolve the whole decoder sequence together.

        q_current_all_dec = q_embedded_dec
        p_current_all_dec = p_initial_dec

        # This requires _calculate_potential_energy_and_forces_dec and _symplectic_step_dec
        # to handle all decoder tokens simultaneously.

        # --- REVISED DECODER EVOLUTION (to process all decoder tokens together) ---
        # We need a revised _calculate_potential_energy_and_forces_dec that processes all q_dec tokens
        # and a revised _symplectic_step_dec.
        
        # For now, let's keep the per-token evolution for illustration of concept, then discuss batching it.
        # This is less efficient and less "Hamiltonian" for the whole decoder sequence.
        
        # The Hamiltonian part is tricky if each token's V depends on attention.
        # Let's assume for a moment that the "Hamiltonian evolution" applies to an *internal state* per decoder step,
        # and this state is then projected to logits. This is a conceptual shift.

        # Simpler conceptual model for now:
        # Decoder is more like a standard RNN/Transformer decoder but with Hamiltonian-like blocks
        # This means `num_hamiltonian_steps` would apply *per decoder output token step*.
        
        # Let's assume a more standard decoder structure for generating output sequence logits,
        # where the Hamiltonian dynamics are part of *how each token's representation is updated*.
        
        # Iterating over decoder sequence length (standard for autoregressive models)
        # In teacher forcing, we have all dec_input_ids.
        q_current_dec_states = q_embedded_dec # (batch, seq_len_dec, d_embed)
        p_current_dec_states = p_initial_dec # (batch, seq_len_dec, d_embed)

        # We need to apply Hamiltonian evolution to q_current_dec_states & p_current_dec_states
        # The potential for q_dec_i will depend on q_dec_i and attention to encoder_q_outputs.
        
        # ---- New approach for decoder Hamiltonian evolution ----
    def _calculate_full_dec_potential_and_forces(self, q_all_dec_state, encoder_q_outputs, dec_padding_mask, enc_padding_mask):
        # q_all_dec_state: (batch, seq_len_dec, d_embedding)
        batch_size, seq_len_dec, _ = q_all_dec_state.shape
        
        q_all_dec_for_V = q_all_dec_state.detach().requires_grad_(True)
        
        # Calculate attended context for ALL decoder q tokens at once
        # cross_attention expects query (B, S_dec, D), key (B, S_enc, D), value (B, S_enc, D)
        attended_encoder_contexts, _ = self.cross_attention(
            q_all_dec_for_V, encoder_q_outputs, encoder_q_outputs, enc_padding_mask
        ) # (batch, seq_len_dec, d_embedding)

        potential_mlp_input = torch.cat([q_all_dec_for_V, attended_encoder_contexts.detach()], dim=-1)
        
        with torch.enable_grad():
            V_per_dec_token = self.potential_mlp(potential_mlp_input) # (batch, seq_len_dec, 1)
            if dec_padding_mask is not None:
                V_per_dec_token = V_per_dec_token * dec_padding_mask.unsqueeze(-1).float()
            V_total_scalar = V_per_dec_token.sum()
            forces = -torch.autograd.grad(V_total_scalar, q_all_dec_for_V, create_graph=True, retain_graph=True)[0]

        if dec_padding_mask is not None:
            forces = forces * dec_padding_mask.unsqueeze(-1).float()
        if self.force_clip_value is not None:
            forces = torch.clamp(forces, -self.force_clip_value, self.force_clip_value)
        return V_per_dec_token.sum(dim=(1,2)).detach(), forces

    def _symplectic_step_full_dec(self, q_all_current_dec, p_all_current_dec,
                                 encoder_q_outputs, dec_padding_mask, enc_padding_mask):
        masses = self.get_masses().to(q_all_current_dec.device) # (1, seq_len_dec, 1)

        _, forces_at_q_current = self._calculate_full_dec_potential_and_forces(
            q_all_current_dec, encoder_q_outputs, dec_padding_mask, enc_padding_mask
        )
        p_half = p_all_current_dec + (self.h_step_integrator / 2.0) * forces_at_q_current
        
        q_delta = self.h_step_integrator * (p_half / masses)
        if dec_padding_mask is not None: q_delta = q_delta * dec_padding_mask.unsqueeze(-1).float()
        q_all_next_dec = q_all_current_dec + q_delta
        
        _, forces_at_q_next = self._calculate_full_dec_potential_and_forces(
            q_all_next_dec, encoder_q_outputs, dec_padding_mask, enc_padding_mask
        )
        p_all_next_dec = p_half + (self.h_step_integrator / 2.0) * forces_at_q_next
        
        if dec_padding_mask is not None: p_all_next_dec = p_all_next_dec * dec_padding_mask.unsqueeze(-1).float()
        return q_all_next_dec, p_all_next_dec

    # Resuming original forward for Decoder
    # forward(self, dec_input_ids, dec_padding_mask, encoder_q_outputs, enc_padding_mask):
        # ... (q_embedded_dec and p_initial_dec setup)
        q_current_all_dec = q_embedded_dec
        p_current_all_dec = p_initial_dec

        for _ in range(self.num_hamiltonian_steps):
            q_current_all_dec, p_current_all_dec = self._symplectic_step_full_dec(
                q_current_all_dec, p_current_all_dec,
                encoder_q_outputs, dec_padding_mask, enc_padding_mask
            )
        
        # Final q states of decoder are used for logits
        final_decoder_q_states = q_current_all_dec
        output_logits = self.output_layer_logits(final_decoder_q_states) # (batch, seq_len_dec, vocab_size)
        return output_logits


class EncoderDecoderHamiltonianModel(nn.Module):
    def __init__(self, vocab_size, d_embedding, enc_seq_len, dec_seq_len, 
                 d_hidden_potential_enc, d_hidden_potential_dec,
                 num_ham_steps_enc, num_ham_steps_dec, h_step_integrator, 
                 delta_t_momentum, force_clip_value, pad_idx, num_attn_heads=4):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = HamiltonianEncoder(
            vocab_size=vocab_size, d_embedding=d_embedding, sequence_length=enc_seq_len,
            d_hidden_potential=d_hidden_potential_enc, num_hamiltonian_steps=num_ham_steps_enc,
            h_step_integrator=h_step_integrator, delta_t_momentum=delta_t_momentum,
            force_clip_value=force_clip_value, pad_idx=pad_idx
        )
        self.decoder = HamiltonianDecoder(
            vocab_size=vocab_size, d_embedding=d_embedding, sequence_length=dec_seq_len,
            d_hidden_potential=d_hidden_potential_dec, num_hamiltonian_steps=num_ham_steps_dec,
            h_step_integrator=h_step_integrator, # delta_t_momentum not used by decoder p_initial
            force_clip_value=force_clip_value, pad_idx=pad_idx,
            d_encoder_output=d_embedding, num_attn_heads=num_attn_heads
        )

    def forward(self, enc_input_ids, dec_input_ids):
        # Create padding masks based on pad_idx
        enc_padding_mask = (enc_input_ids != self.pad_idx)
        dec_padding_mask = (dec_input_ids != self.pad_idx)

        # Encoder pass
        # encoder_q_outputs, encoder_p_outputs = self.encoder(enc_input_ids, enc_padding_mask)
        # For simplicity, let's assume decoder only needs encoder_q_outputs for attention
        encoder_q_outputs, _ = self.encoder(enc_input_ids, enc_padding_mask)

        # Decoder pass
        decoder_output_logits = self.decoder(
            dec_input_ids, dec_padding_mask,
            encoder_q_outputs, enc_padding_mask
        )
        return decoder_output_logits