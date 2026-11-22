# Reproduce

Run from the unpacked `group_xxx/` folder (weights are in `code/weights/`).

```bash
pip install -r code/requirements.txt
# fast entry used for the submission: writes group_new/ and group_new.zip
python code/submit_v3.py --index /path/to/test_input --out group_new --steps 100 --lr 0.08 --offaxis-model bbox_eye_exact --offaxis-initial sart
# TA template entry (create_predictor implemented, serial and slower)
python code/inference.py --index /path/to/test.json --out reproduced --group group_new --weights code/weights/pretrained_model.pth
# format check
python code/check_submission.py --index /path/to/test.json --submission index.json
```

`code/metrics.json` holds the per-sample self-scores and per-eye off-axis K of this run.
No training is performed; EyeReal source (AGPL-3.0) is in `code/EyeReal/`.
