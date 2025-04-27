import json, pickle
from utils.core_utils import *
import random, torch
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


def inference(args):
    """   
        test for a single fold
    """
    
    model_fp = args.model_fp
    fold_name = args.fold_name
    experiment_name = args.experiment_name
    experiment_rp = os.path.join(args.experiment_rp, experiment_name)
    
    output_rps = [os.path.join(experiment_rp, "output", i) for i in ["attn_score", "heatmap", "metrics", "model"]]
    for output_rp in output_rps:
        if not os.path.exists(output_rp):
            os.makedirs(output_rp)
        
    attn_score_fp, _, metric_rp, model_rp = output_rps
    
    slide_prompt_fp = os.path.join(experiment_rp, "input/prompt", "slide_prompts.json")
    patch_prompt_fp = os.path.join(experiment_rp, "input/prompt", "patch_prompts.json")
    split_label_fp = os.path.join(experiment_rp, "input/csv/split", fold_name + '.csv')
           
    n_classes = args.n_classes
    feature_rp = args.feature_rp
    lr = args.learning_rate
    vlm_name = args.vlm_name
    attn_type = getattr(args, 'attn_type', 'default_value')
    num_epochs = args.num_epochs
    
    model_name = f"{vlm_name}_"+"_".join(fold_name.split("/")[-2:])+f"_lr:{lr}_{attn_type}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_split = pd.read_csv(split_label_fp)
    
    train_data_fps = [os.path.join(feature_rp, i.split("/")[-1].replace(".svs", ".pkl")) for i in list(df_split[df_split["type"]=="train"]["data_path"])]
    train_labels = list(df_split[df_split["type"]=="train"]["label"])

    test_data_fps = [os.path.join(feature_rp, i.split("/")[-1].replace(".svs", ".pkl")) for i in list(df_split[df_split["type"]=="test"]["data_path"])]
    test_labels = list(df_split[df_split["type"]=="test"]["label"])
    
    val_data_fps = [os.path.join(feature_rp, i.split("/")[-1].replace(".svs", ".pkl")) for i in list(df_split[df_split["type"]=="val"]["data_path"])]
    val_labels = list(df_split[df_split["type"]=="val"]["label"])  
      
    print(f"test on {len(test_data_fps)} samples")
    
    loss_fn = nn.CrossEntropyLoss()
    
    with open(slide_prompt_fp, 'r', encoding='utf-8') as file:
        slide_level_prompts = json.load(file)
    slide_level_prompts_ = [f"{k} {v}" for k, v in slide_level_prompts.items()]
    
    with open(patch_prompt_fp, 'r', encoding='utf-8') as file:
        patch_level_prompts = json.load(file)
    patch_level_prompts_ = [f"{subk} {subv}" for k in patch_level_prompts for subk, subv in patch_level_prompts[k].items()]
    
    if args.vlm_name == "clip":
        from trainers import top_clip as top
        from clip import clip
        clip_model, _ = clip.load(clip_base_model, device=device)
        
    elif args.vlm_name == "plip":
        from trainers import top_plip as top
        from transformers import CLIPModel
        clip_model = CLIPModel.from_pretrained("vinid/plip").to(device)
        
    elif args.vlm_name == "quilt1m":
        from trainers import ConcepPath as top
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        clip_model = clip_model.to(device)
        
    model = top.ConcepPath(
        slide_prompt=slide_level_prompts_,
        patch_prompt=patch_level_prompts_,
        clip_model=clip_model,
        loss_func=loss_fn,
        num_patch_prompt=args.num_patch_prompt,
        n_ctx=args.n_ctx,
        is_shared=args.is_shared,
        n_ddp=args.n_ddp,
        n_classes=args.n_classes,
        orth_ratio=args.orth_ratio,
    ).to(device)
    
    if model_fp != "":
        weights = torch.load(model_fp)
        # Filter and load weights safely
        model.load_state_dict({k: v for k, v in weights.items() if k in model.state_dict()}, strict=False)
        print("The pre-trained model is loaded")
    
    top_transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.type(torch.float32)
    ])
    
    test_dataset = TOPDataset(test_data_fps, test_labels, top_transform)
    print("Testing on {} samples".format(len(test_dataset)))
    test_loader = get_split_loader(test_dataset)
    
    with torch.no_grad():
        model.eval()
        test_loss, test_acc, test_micro_f1, test_macro_f1, test_micro_auc, test_macro_auc, test_avg_sensitivity, test_avg_specificity, test_patient_results, all_preds, all_labels = test(
            model, test_loader, n_classes, test_data_fps, attn_score_fp, vlm_name, return_preds=True
        )
    
    print(f"test_acc: {test_acc}, test_micro_f1: {test_micro_f1}, test_macro_f1: {test_macro_f1}, test_micro_auc: {test_micro_auc}, test_macro_auc: {test_macro_auc}, test_avg_sensitivity: {test_avg_sensitivity}, test_avg_specificity: {test_avg_specificity}")

    # ===== Confusion Matrix =====
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save as image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {fold_name}")
    plt.savefig(os.path.join(metric_rp, f"confusion_matrix_{fold_name}.png"))
    plt.close()

    # Save as CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(metric_rp, f"confusion_matrix_{fold_name}.csv"), index=False)
