python code/check_env.py --out outputs/env_check.json && \
python code/inspect_data.py --input data/pretrain_data.pt --out outputs/data_audit_pretrain.json && \
python code/inspect_data.py --input data/finetune_data.pt --out outputs/data_audit_finetune.json && \
python code/inspect_data.py --input data/candidate_data.pt --out outputs/data_audit_candidate.json && \
python code/make_splits.py --input data/finetune_data.pt --strategy stratified --seeds 3 --out outputs/splits && \
python code/train_pipeline.py --mode baseline --outdir outputs/baseline && \
python code/train_pipeline.py --mode weighted --outdir outputs/weighted && \
python code/train_pipeline.py --mode focal --outdir outputs/focal && \
python code/train_pipeline.py --mode pretrained --outdir outputs/pretrained && \
python code/summarize_results.py --inputs outputs/baseline outputs/weighted outputs/focal outputs/pretrained --report_dir report --image_dir report/images
