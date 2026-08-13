from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from clat.data import DataItem
from clat.encoder import load_encoder
from clat.utils import CLATOutput


class KnowledgeGuideLoss(nn.Module):
    def __init__(
        self, knowledge_embeds: torch.Tensor, eps=1e-8, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.knowledge_embeds_T = knowledge_embeds.T
        self.eps = eps

    def forward(self, inputs, targets):
        scores = inputs @ self.knowledge_embeds_T.to(inputs.device)
        gt = torch.arange(targets.size(-1), dtype=torch.long, device=targets.device)
        gt = gt.unsqueeze(0).expand(scores.shape[0], scores.shape[1])
        loss = self.loss(scores, gt)
        loss = torch.mean(
            torch.mean(loss * targets, dim=-1)
            / (torch.sum(targets, dim=-1) + self.eps),
            dim=-1,
        )
        return loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: Optional[list[int]] = None,
        beta: float = 0.999,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if class_counts:
            counts = torch.tensor(class_counts, dtype=torch.float)
            effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
            weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-8)
            weights = weights / weights.sum() * len(class_counts)
            self.register_buffer("class_weights", weights)
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(inputs.device)[targets]
        return loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, inputs, targets):
        targets = targets.float()
        prob_pos = torch.sigmoid(inputs)
        prob_neg = 1.0 - prob_pos
        if self.clip and self.clip > 0:
            prob_neg = torch.clamp(prob_neg + self.clip, max=1.0)

        loss_pos = targets * torch.log(torch.clamp(prob_pos, min=self.eps))
        loss_neg = (1 - targets) * torch.log(torch.clamp(prob_neg, min=self.eps))

        pt = prob_pos * targets + prob_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss = (loss_pos + loss_neg) * ((1 - pt) ** gamma)
        return -loss.mean()


class PrototypeDiversityLoss(nn.Module):
    def forward(self, lesion_proto_tokens):
        if lesion_proto_tokens is None or lesion_proto_tokens.size(2) <= 1:
            if lesion_proto_tokens is None:
                return torch.tensor(0.0)
            return lesion_proto_tokens.sum() * 0.0

        proto_tokens = F.normalize(lesion_proto_tokens, dim=-1)
        sim = torch.matmul(proto_tokens, proto_tokens.transpose(-1, -2))
        k = sim.size(-1)
        off_diag = ~torch.eye(k, dtype=torch.bool, device=sim.device)
        return sim[..., off_diag].pow(2).mean()


class CLAT(LightningModule):
    def __init__(
        self,
        disease_names: list[str],
        lesion_names: list[str],
        img_size: int = 224,
        arch_name: str = "cait_s24_224_concept",
        pretrained: bool = True,
        disease_loss_weight: float = 1.0,
        lesion_loss_weight: float = 0.6,
        KG_loss_weight: float = 0.4,
        proto_diversity_loss_weight: float = 0.0,
        counterfactual_loss_weight: float = 0.0,
        with_EK: bool = False,
        lesion_proto_count: int = 1,
        disease_loss_type: Literal["ce", "cb_focal"] = "ce",
        disease_class_counts: Optional[list[int]] = None,
        focal_gamma: float = 2.0,
        cb_beta: float = 0.999,
        lesion_loss_type: Literal["soft_margin", "asymmetric"] = "soft_margin",
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 1.0,
        asl_clip: float = 0.05,
        counterfactual_margin: float = 0.10,
        counterfactual_prob: float = 0.5,
        training_int_prob: Optional[float] = None,
        training_int_milestone: int = 0,
        eval_int: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_disease = len(disease_names)
        self.num_lesions = len(lesion_names)
        self.disease_names = disease_names
        self.lesion_names = lesion_names
        self.img_size = img_size
        self.training_int_prob = training_int_prob
        self.training_int_milestone = training_int_milestone
        self.eval_int = eval_int

        self.model = load_encoder(
            arch_name,
            pretrained=pretrained,
            num_classes=self.num_disease,
            num_lesions=self.num_lesions,
            img_size=img_size,
            lesion_proto_count=lesion_proto_count,
        )
        if "No DR" in disease_names:
            knowledge_embeds_path = "data/FLAIR_DR_with_EK.pt"
        elif "No RAO" in disease_names:
            knowledge_embeds_path = "data/FLAIR_RAO_with_EK.pt"
        if not with_EK:
            knowledge_embeds_path = knowledge_embeds_path.replace(
                "with_EK", "without_EK"
            )
        knowledge_embeds = torch.load(knowledge_embeds_path)
        self.token2concept = nn.Linear(self.model.embed_dim, knowledge_embeds.shape[1])
        self.patch_size = self.model.patch_embed.patch_size

        self.disease_loss_weight = disease_loss_weight
        self.lesion_loss_weight = lesion_loss_weight
        self.KG_loss_weight = KG_loss_weight
        self.proto_diversity_loss_weight = proto_diversity_loss_weight
        self.counterfactual_loss_weight = counterfactual_loss_weight
        self.counterfactual_margin = counterfactual_margin
        self.counterfactual_prob = counterfactual_prob

        self.loss_disease = (
            ClassBalancedFocalLoss(
                disease_class_counts,
                beta=cb_beta,
                gamma=focal_gamma,
            )
            if disease_loss_type == "cb_focal"
            else nn.CrossEntropyLoss()
        )
        self.loss_lesion = (
            AsymmetricLoss(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip,
            )
            if lesion_loss_type == "asymmetric"
            else nn.MultiLabelSoftMarginLoss()
        )
        self.loss_knowledge_guide = KnowledgeGuideLoss(knowledge_embeds)
        self.loss_proto_diversity = PrototypeDiversityLoss()

    def forward(self, x, return_attn=False, **kwargs) -> CLATOutput:
        return self.model(x, return_attn=return_attn, **kwargs)

    def counterfactual_loss(self, images, disease_lbls, lesion_lbls, output):
        if (
            not self.training
            or self.counterfactual_loss_weight <= 0
            or self.counterfactual_prob <= 0
        ):
            return output.disease_logits.sum() * 0.0

        positive_mask = lesion_lbls.float() > 0.5
        sample_mask = (
            torch.rand(images.size(0), device=images.device) < self.counterfactual_prob
        ) & positive_mask.any(dim=1)
        if not sample_mask.any():
            return output.disease_logits.sum() * 0.0

        selected = torch.multinomial(positive_mask[sample_mask].float(), 1).squeeze(1)
        lesion_token_mask = torch.zeros_like(lesion_lbls.float())
        lesion_token_mask[sample_mask, selected] = 1.0

        cf_output = self(
            images,
            lesion_token_mask=lesion_token_mask,
        )
        original_prob = torch.softmax(output.disease_logits.detach(), dim=-1)
        cf_prob = torch.softmax(cf_output.disease_logits, dim=-1)

        original_gt_prob = original_prob.gather(1, disease_lbls.unsqueeze(1)).squeeze(1)
        cf_gt_prob = cf_prob.gather(1, disease_lbls.unsqueeze(1)).squeeze(1)
        drop = original_gt_prob - cf_gt_prob
        return F.relu(self.counterfactual_margin - drop[sample_mask]).mean()

    def shared_step(
        self, batch: DataItem, stage: Literal["train", "val", "test"]
    ) -> dict[str, Any]:
        images, disease_lbls, lesion_lbls = (
            batch.image,
            batch.disease_lbls,
            batch.lesion_lbls,
        )
        bs = images.shape[0]

        output: CLATOutput
        # forward
        if (
            self.training
            and self.training_int_prob
            and self.training_int_milestone is not None
            and self.current_epoch >= self.training_int_milestone
        ):
            output = self(
                images, int_prob=self.training_int_prob, lesion_lbls=lesion_lbls
            )
        else:
            output = self(images, return_attn=stage == "test")

        # loss
        disease_loss = (
            self.loss_disease(output.disease_logits, disease_lbls)
            if self.disease_loss_weight > 0
            else 0
        )
        lesion_loss = (
            self.loss_lesion(output.lesion_logits, lesion_lbls)
            if self.lesion_loss_weight > 0
            else 0
        )
        KG_loss = (
            self.loss_knowledge_guide(
                self.token2concept(output.lesion_tokens), lesion_lbls
            )
            if self.KG_loss_weight > 0
            else 0
        )
        proto_diversity_loss = (
            self.loss_proto_diversity(output.lesion_proto_tokens)
            if self.proto_diversity_loss_weight > 0
            else 0
        )
        counterfactual_loss = (
            self.counterfactual_loss(images, disease_lbls, lesion_lbls, output)
            if self.counterfactual_loss_weight > 0
            else 0
        )

        loss = (
            self.disease_loss_weight * disease_loss
            + self.lesion_loss_weight * lesion_loss
            + self.KG_loss_weight * KG_loss
            + self.proto_diversity_loss_weight * proto_diversity_loss
            + self.counterfactual_loss_weight * counterfactual_loss
        )

        self.log(
            f"loss/{stage}_loss", loss, prog_bar=True, batch_size=bs, sync_dist=True
        )
        self.log_dict(
            {
                f"loss/{stage}_disease_loss": disease_loss,
                f"loss/{stage}_lesion_loss": lesion_loss,
                f"loss/{stage}_KG_loss": KG_loss,
                f"loss/{stage}_proto_diversity_loss": proto_diversity_loss,
                f"loss/{stage}_counterfactual_loss": counterfactual_loss,
            },
            prog_bar=False,
            batch_size=bs,
            sync_dist=True,
        )

        ret_dict = {"loss": loss, **output._asdict()}

        return ret_dict

    def get_explanations(self, image):
        pred = self(image, return_attn=True)
        disease_logits = pred.disease_logits
        lesion_logits = pred.lesion_logits
        cross_attn_maps = pred.cross_attn_maps
        disease_confidence, disease_type = torch.max(
            torch.softmax(disease_logits, dim=-1), dim=-1
        )
        disease_type = disease_type.item()
        disease_confidence = disease_confidence.item()
        lesion_confidence = (
            torch.sigmoid(lesion_logits).squeeze().detach().cpu().numpy()
        )
        contributions = (
            cross_attn_maps.sum(dim=1)
            .squeeze()
            .softmax(dim=-1)[disease_type]
            .detach()
            .cpu()
            .numpy()
        )

        expl = self._expl(
            disease_type, disease_confidence, lesion_confidence, contributions
        )

        return expl, pred

    def _expl(
        self, disease_type, disease_confidence, lesion_confidence, contributions
    ) -> str:
        expl = [
            f"Disease diagnosis: {self.disease_names[disease_type]}({disease_confidence:.2%})\n"
        ]
        expl.append("Lesions discovery:\n")
        exist_lesions, absent_lesions = [], []
        for i in range(self.num_lesions):
            if lesion_confidence[i] > 0.5:
                exist_lesions.append(f"{self.lesion_names[i]}({contributions[i]:.2%})")
            else:
                absent_lesions.append(f"{self.lesion_names[i]}({contributions[i]:.2%})")
            expl.append(f"{self.lesion_names[i]}({lesion_confidence[i]:.2%})\n")
        expl.append(f"This is {self.disease_names[disease_type]}. The existence of ")
        for e in exist_lesions:
            if e == exist_lesions[0]:
                expl.append(f"{e}, ")
            elif e == exist_lesions[-1]:
                expl.append(f"and {e}")
            else:
                expl.append(f"{e}, ")
        expl.append("  and the absent of ")
        for a in absent_lesions:
            if a == absent_lesions[0]:
                expl.append(f"{a}, ")
            elif a == absent_lesions[-1]:
                expl.append(f"and {a} confirms this diagnosis.")
            else:
                expl.append(f"{a}, ")
        return "".join(expl)

    def intervene(self, image, intervene_lesion_idx, interven_lesion_probs):
        pred = self(
            image,
            return_attn=True,
            intervene_sample_idx=0,
            intervene_cpt_idx=intervene_lesion_idx,
            lesion_lbls=interven_lesion_probs,
        )
        disease_logits = pred.disease_logits
        lesion_logits = pred.lesion_logits
        cross_attn_maps = pred.cross_attn_maps
        disease_confidence, disease_type = torch.max(
            torch.softmax(disease_logits, dim=-1), dim=-1
        )
        disease_type = disease_type.item()
        disease_confidence = disease_confidence.item()
        lesion_confidence = (
            torch.sigmoid(lesion_logits).squeeze().detach().cpu().numpy()
        )
        contributions = (
            cross_attn_maps.sum(dim=1)
            .squeeze()
            .softmax(dim=-1)[disease_type]
            .detach()
            .cpu()
            .numpy()
        )

        expl = self._expl(
            disease_type, disease_confidence, lesion_confidence, contributions
        )

        return expl, pred

    def training_step(self, batch):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch):
        return self.shared_step(batch, stage="val")

    def test_step(self, batch):
        return self.shared_step(batch, stage="test")
