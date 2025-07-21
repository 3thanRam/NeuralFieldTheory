# common/NeuralFieldNetwork.py - The core model architecture.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Classes (Unchanged but included for completeness) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding))
        pe = torch.zeros(max_len, d_embedding)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class LearnableFrFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1) * 0.01)

    def forward(self, q, p):
        c, s = torch.cos(self.alpha), torch.sin(self.alpha)
        return q * c + p * s, -q * s + p * c

    def inverse(self, q, p):
        c, s = torch.cos(self.alpha), torch.sin(self.alpha)
        return q * c - p * s, q * s + p * c

class InvertibleTransform(nn.Module):
    def __init__(self, d_embedding, d_hidden):
        super().__init__()
        layer = lambda: nn.Sequential(nn.Linear(d_embedding, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_embedding))
        self.s1, self.t1, self.s2, self.t2 = layer(), layer(), layer(), layer()

    def forward(self, q, p):
        log_s1 = self.s1(p); t1 = self.t1(p)
        q_intermediate = torch.exp(log_s1) * q + t1
        log_s2 = self.s2(q_intermediate); t2 = self.t2(q_intermediate)
        return q_intermediate, torch.exp(log_s2) * p + t2, log_s1, log_s2

    def inverse(self, a_real, a_imag):
        log_s2 = self.s2(a_real); t2 = self.t2(a_real)
        p_intermediate = (a_imag - t2) * torch.exp(-log_s2)
        log_s1 = self.s1(p_intermediate); t1 = self.t1(p_intermediate)
        return (a_real - t1) * torch.exp(-log_s1), p_intermediate

class QuadraticInteraction(nn.Module):
    def __init__(self, d_embedding, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(d_embedding,d_embedding,kernel_size,padding=(kernel_size-1)//2,groups=d_embedding,bias=False)
    def forward(self, x1, x2):
        potential = self.conv(x1.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.einsum('bsd,bsd->b', x2, potential)

class SubspaceInteraction(nn.Module):
    def __init__(self,d,h,ns=4,sd=64,nh=4,**k):
        super().__init__();self.ns,self.sd=ns,sd
        self.q_projs=nn.ModuleList([nn.Linear(d,sd,bias=False)for _ in range(ns)])
        self.p_projs=nn.ModuleList([nn.Linear(d,sd,bias=False)for _ in range(ns)])
        self.attn=nn.MultiheadAttention(sd,nh,batch_first=True,dropout=0.1);self.norm=nn.LayerNorm(sd)
        self.mlp=nn.Sequential(nn.Linear(ns*2*sd,h),nn.GELU(),nn.Linear(h,1))
    def forward(self,q,p):
        bs,sl,_=q.shape;qs=torch.stack([proj(q)for proj in self.q_projs],dim=2);ps=torch.stack([proj(p)for proj in self.p_projs],dim=2)
        a=torch.cat([qs,ps],dim=2).view(bs*sl,self.ns*2,self.sd);o,_=self.attn(a,a,a);p=self.norm(a+o);e=self.mlp(p.flatten(1)).view(bs,sl).sum(1);return e

class HamiltonianBlock(nn.Module):
    def __init__(self,d,h,**k):
        super().__init__();self.nq,self.np=nn.LayerNorm(d),nn.LayerNorm(d)
        self.lq,self.lp=nn.Parameter(torch.randn(d)),nn.Parameter(torch.randn(d));ks=k.get('kernel_size',3)
        self.qq,self.pp,self.qp=QuadraticInteraction(d,ks),QuadraticInteraction(d,ks),QuadraticInteraction(d,ks)
        self.interact=SubspaceInteraction(d,h,**k);self.offset=nn.Parameter(torch.randn(1))
    def forward(self,q,p):
        qn,pn=self.nq(q),self.np(p)
        Hl=torch.einsum('bsd,d->b',qn,self.lq)+torch.einsum('bsd,d->b',pn,self.lp)
        Hq=self.qq(qn,qn)+self.pp(pn,pn)+self.qp(qn,pn);Hn=self.interact(qn,pn);return self.offset+Hl+Hq+Hn
    def gradients(self,q,p):
        q.requires_grad_(True);p.requires_grad_(True);H=self.forward(q,p).sum()
        return torch.autograd.grad(H,(q,p),grad_outputs=torch.ones_like(H),create_graph=True)

class HamiltonianCore(nn.Module):
    def __init__(self, d, h, nb, **k):
        super().__init__()
        self.num_blocks = nb
        self.timestep = k.get('timestep', 0.1)
        self.pos_encoder = PositionalEncoding(d)
        self.q_shift = nn.Parameter(torch.zeros(d))
        self.dropout = nn.Dropout(k.get('dropout', 0.1))
        self.momentum_net = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, d))
        self.frft_transform = LearnableFrFT()
        self.coord_transform = InvertibleTransform(d, h // 2)
        self.q_norm = nn.LayerNorm(d)
        self.p_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([HamiltonianBlock(d, h, **k) for _ in range(nb)])
        self.norm = nn.LayerNorm(d)

    def leapfrog(self, q, p, block):
        # ... (This method is correct) ...
        dH_dp, dH_dq = block.gradients(q, p); dH_dp, dH_dq = torch.clamp(dH_dp, -1, 1), torch.clamp(dH_dq, -1, 1)
        p_half = p - (self.timestep / 2) * dH_dq; q_new = q + self.timestep * dH_dp
        _, dH_dq_new = block.gradients(q_new, p_half); p_new = p_half - (self.timestep / 2) * torch.clamp(dH_dq_new, -1, 1)
        return q_new, p_new

    # --- THIS IS THE CORRECTED METHOD SIGNATURE ---
    def forward(self, initial_vectors, return_internals=False):
        q_initial = self.dropout(self.pos_encoder(initial_vectors) + self.q_shift)
        dq = torch.zeros_like(q_initial)
        dq[:, 1:] = q_initial[:, 1:] - q_initial[:, :-1]
        p_initial = self.momentum_net(torch.cat([q_initial, dq], dim=-1))
        q_rot, p_rot = self.frft_transform(q_initial, p_initial)
        Q_initial, P_initial, log_s1, log_s2 = self.coord_transform(self.q_norm(q_rot), self.p_norm(p_rot))
        Q, P = Q_initial, P_initial
        reversibility_losses, energies_initial, energies_final = [], [], []

        with torch.enable_grad():
            for block in self.blocks:
                q_start, p_start = Q.detach(), P.detach()
                if return_internals:
                    energies_initial.append(block(q_start, p_start))
                
                q_new, p_new = self.leapfrog(q_start, p_start, block)
                
                if return_internals:
                    energies_final.append(block(q_new, p_new))
                
                Q, P = q_new.detach(), p_new.detach()
                
                if self.training:
                    q_rev_start, p_rev_start = Q, P
                    dH_dp_rev, dH_dq_rev = block.gradients(q_rev_start, p_rev_start)
                    p_half_rev = p_rev_start + (self.timestep / 2) * dH_dq_rev
                    q_new_rev = q_rev_start - self.timestep * dH_dp_rev
                    _, dH_dq_new_rev = block.gradients(q_new_rev, p_half_rev)
                    p_new_rev = p_half_rev + (self.timestep / 2) * dH_dq_new_rev
                    reversibility_losses.append(F.mse_loss(q_new_rev, q_start) + F.mse_loss(p_new_rev, p_start))

        reversibility_loss = sum(reversibility_losses) / len(reversibility_losses) if reversibility_losses else torch.tensor(0.)
        q_transformed, p_transformed = self.coord_transform.inverse(Q, P)
        q_final, p_final = self.frft_transform.inverse(q_transformed, p_transformed)
        hidden_state = self.norm(q_final + q_initial)

        if return_internals:
            internals = {
                'hamiltonian': (Q_initial, P_initial, Q, P),
                'jacobian': (log_s1, log_s2),
                'consistency': (q_final, p_final),
                'reversibility_loss': reversibility_loss,
                'energy': (energies_initial, energies_final)
            }
            return hidden_state, internals
        return hidden_state


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, d_hidden_dim, num_blocks, **kwargs):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.core = HamiltonianCore(
            d=embed_dim,
            h=d_hidden_dim,
            nb=num_blocks,
            **kwargs
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, return_internals=False):
        initial_vectors = self.token_embedding(x)
        output = self.core(initial_vectors, return_internals)
        if return_internals:
            return self.lm_head(output[0]), output[1]
        return self.lm_head(output)

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.8, top_k=20):
        self.eval(); ids = start_ids
        for _ in range(max_new_tokens):
            with torch.enable_grad():
                logits, _ = self.forward(ids, return_internals=True)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat((ids, next_id), dim=1)
            if next_id.item() == getattr(self, 'eos_token_id', -1):
                break
        return ids

class DynamicNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps; self.last_mean = None; self.last_std = None
    def forward(self, x):
        self.last_mean = x.mean(dim=-1, keepdim=True)
        self.last_std = x.std(dim=-1, keepdim=True)
        return (x - self.last_mean) / (self.last_std + self.eps)
    def inverse(self, x_norm):
        if self.last_mean is None: raise RuntimeError("Forward pass must be run before inverse.")
        return x_norm * (self.last_std + self.eps) + self.last_mean

class TimeSeriesPredictor(nn.Module):
    def __init__(self, num_input_features, num_output_predictions, embed_dim, d_hidden_dim, num_blocks, **kwargs):
        super().__init__()
        self.normalizer = DynamicNorm()
        self.input_projection = nn.Linear(num_input_features, embed_dim)
        self.core = HamiltonianCore(
            d=embed_dim,
            h=d_hidden_dim,
            nb=num_blocks,
            **kwargs
        )
        self.output_projection = nn.Linear(embed_dim, num_output_predictions)

    def forward(self, x_raw, return_internals=False):
        x_normalized = self.normalizer.forward(x_raw)
        initial_vectors = self.input_projection(x_normalized)
        
        if return_internals:
            hidden_state, internals = self.core(initial_vectors, return_internals=True)
        else:
            hidden_state = self.core(initial_vectors, return_internals=False)

        projected_output_norm = self.output_projection(hidden_state)
        final_predictions = self.normalizer.inverse(projected_output_norm)
        
        if return_internals:
            return final_predictions, internals
        return final_predictions