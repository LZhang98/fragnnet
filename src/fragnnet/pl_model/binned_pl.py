from fragnnet.model.loss import sparse_cosine_distance_binned
from fragnnet.pl_model import SpectrumPL


class BinnedPL(SpectrumPL):
    def _setup_loss_fn(self):
        # binned cosine distance
        def cos_dist_fn(
            true_mzs, true_logprobs, true_batch_idxs, pred_mzs, pred_logprobs, pred_batch_idxs
        ):
            return sparse_cosine_distance_binned(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                log_distance=(self.hparams.loss_type == "log_cosine_distance"),
            )

        assert self.hparams.loss_type == "cosine_distance", self.hparams.loss_type
        assert self.hparams.sparse_cosine_similarity
        self.binned_loss = True

        def _loss_fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            **kwargs,
        ):
            spec_cd = cos_dist_fn(
                true_mzs, true_logprobs, true_batch_idxs, pred_mzs, pred_logprobs, pred_batch_idxs
            )
            loss = spec_cd
            loss_d = {"loss": loss, "spec_cd": spec_cd}
            return loss_d

        self.loss_fn = _loss_fn
        # flag metrics for tracking
        loss_names = ["loss", "spec_cd"]
        self.metric_names.update(loss_names)
