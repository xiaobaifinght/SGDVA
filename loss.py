import torch
import torch.nn as nn
import torch.nn.functional as F
class Loss(nn.Module):
   
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask
    def __init__(self, batch_size,   temperature_l, temperature_f,walk_steps,   device):  
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.temperature_f = temperature_f       
        self.softmax = nn.Softmax(dim=1)
        self.temperature_l = temperature_l

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.kernel_temp = 0.12
        self.walk_steps = walk_steps
        self.alpha = 0.6
        self.kl_temp = 1.2
        self.eps = 1e-8
    def forward_robust_instance_contrast(self, h_consensus, h_view_for_pos):
    
        with torch.no_grad(): 
            target_relation = self._calculate_high_order_relation(h_consensus)

        q = F.normalize(h_consensus, p=2, dim=1)
        k = F.normalize(h_view_for_pos, p=2, dim=1)
    
        sim_matrix = torch.matmul(q, k.t()) / self.temperature_f
        log_predicted_prob = F.log_softmax(sim_matrix, dim=1)
        
        target_prob = F.softmax(target_relation / self.kl_temp, dim=1)
        
        loss = F.kl_div(log_predicted_prob, target_prob, reduction='batchmean')      
        return  loss
    
    def forward_instance(self, h_consensus, h_view, reduction='mean'):
        with torch.no_grad():
            relation_teacher = self._calculate_high_order_relation(h_consensus)

        z_student = F.normalize(h_view, p=2, dim=1)
        dist_sq_student = (2 - 2 * torch.matmul(z_student, z_student.t())).clamp(min=0.)
        G_student = torch.exp(-dist_sq_student / (self.kernel_temp + self.eps))
        G_student = G_student / (G_student.sum(dim=1, keepdim=True) + self.eps)
   
        with torch.no_grad():
            if self.walk_steps > 1:
                M_t_student = torch.matrix_power(G_student, self.walk_steps)
            else:
                M_t_student = G_student
            I = torch.eye(G_student.shape[0], device=G_student.device)
            G_robust_student = I * self.alpha + M_t_student * (1 - self.alpha)
            relation_student = G_robust_student / (G_robust_student.sum(dim=1, keepdim=True) + self.eps)
        
        log_p_student = F.log_softmax(relation_student / self.kl_temp, dim=1)
        p_teacher = F.softmax(relation_teacher / self.kl_temp, dim=1)
        
        kl_div = torch.sum(p_teacher * (torch.log(p_teacher + self.eps) - log_p_student), dim=1)
        
        if reduction == 'mean':
            return kl_div.mean()
        elif reduction == 'none':
            return kl_div
    
    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        S_1 = S.repeat(2, 2)
        all_one = torch.ones(self.batch_size*2, self.batch_size*2).to('cuda')
        S_2 = all_one - S_1
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_l
        sim1 = torch.multiply(sim, S_2)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

       
    def _calculate_high_order_relation(self, features):
        z = F.normalize(features, p=2, dim=1)
        dist_matrix_sq = (2 - 2 * torch.matmul(z, z.t())).clamp(min=0.)
        G = torch.exp(-dist_matrix_sq / (self.kernel_temp + self.eps))
        G = G / (G.sum(dim=1, keepdim=True) + self.eps)

        if self.walk_steps > 1:
            G = torch.matrix_power(G, max(1, int(self.walk_steps)))
        I = torch.eye(G.shape[0], device=G.device)
        G_robust = I * self.alpha + G * (1 - self.alpha)
        relation_matrix = G_robust / (G_robust.sum(dim=1, keepdim=True) + self.eps)
        return relation_matrix
    