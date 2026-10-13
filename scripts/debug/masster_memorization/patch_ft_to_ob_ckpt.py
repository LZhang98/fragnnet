# Copy a 4-instrument SPV MASSter checkpoint trained with the retired "FT" label so it runs on
# the current labels. Instrument indices are sorted: trained [FT, NA, Other, QTOF] -> current
# [NA, OB, Other, QTOF], so rows 0/1 swap (OB takes FT's row, NA keeps its own).
import sys, torch as th
src, dst = sys.argv[1:3]
ckpt = th.load(src, map_location="cpu", weights_only=False)
name = "model.context_encoder.instrument_embedding.weight"
for key in ("state_dict", "ema_training_state_dict"):
    if key in ckpt:
        w = ckpt[key][name]
        assert w.shape[0] == 4, w.shape
        ckpt[key][name] = w[[1, 0, 2, 3]].clone()
hp = ckpt["hyper_parameters"]
assert sorted(hp["spec_params"]["inst_types"]) == ["FT", "NA", "Other", "QTOF"]
hp["spec_params"]["inst_types"] = ["OB" if x == "FT" else x for x in hp["spec_params"]["inst_types"]]
hp["inst_type_loss_tol"] = {("OB" if k == "FT" else k): v for k, v in hp["inst_type_loss_tol"].items()}
th.save(ckpt, dst)
print(dst, hp["spec_params"]["inst_types"])
