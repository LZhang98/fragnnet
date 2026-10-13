# Copy a one-class (FT) MASSter checkpoint to a two-class (OB, OB_F) one whose rows are equal,
# so every relabelled spectrum gets the embedding FT gave it: predictions match the old labels.
import sys, torch as th
src, dst = sys.argv[1:3]
ckpt = th.load(src, map_location="cpu", weights_only=False)
for key in ("state_dict", "ema_training_state_dict"):
    if key in ckpt:
        name = "model.context_encoder.instrument_embedding.weight"
        weight = ckpt[key][name]
        assert weight.shape[0] == 1, weight.shape
        ckpt[key][name] = weight.expand(2, -1).clone()
        print(key, tuple(weight.shape), "->", tuple(ckpt[key][name].shape))
th.save(ckpt, dst)
