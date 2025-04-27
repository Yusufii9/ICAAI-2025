from torch.utils.data import Dataset, DataLoader
import torch, os, h5py
import torchvision.transforms as transforms
import pandas as pd
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='seg and patch')

parser.add_argument("--vlm_model", type=str, default="quilt1m", choices=["quilt1m", "plip", "clip"],
                    help="vlm_model")
parser.add_argument('--label_fp', type=str, default="/home/r10user13/TOP/data/datasets/LUNG/subtyping_label.csv",
                    help='file path for label file')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--num_workers', type=int, default=32,
                    help='num_workers')
parser.add_argument('--save_rp', type=str, default="/data2/r10user13/",
                    help='')
parser.add_argument('--base_mag', type=int, default=20,
                    help='')
parser.add_argument('--base_patch_size', type=int, default=448,
                    help='')

class TOPDataset(Dataset):
    def __init__(self, seg_fp, transform):
        self.seg_fp = seg_fp
        self.transform = transform
        
        with h5py.File(seg_fp, 'r') as f:
            self.patches = f['imgs'][:]
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        img_patch = self.patches[idx]  # Directly retrieve stored patch
        
        img_patch = np.moveaxis(img_patch, 0, -1)  # Convert (C, H, W) to (H, W, C)
        img_patch = np.array(img_patch).astype(np.uint8)
        img_patch = transforms.ToPILImage()(img_patch)
        
        return self.transform(img_patch)

if __name__ == "__main__":
    args = parser.parse_args()
    df_seg_fp = pd.read_csv(args.label_fp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vlm_model = args.vlm_model
    if vlm_model == "clip":
        from clip import clip
        clip_model, preprocess = clip.load('ViT-B/16', device=device)
        model = clip_model.visual
    elif vlm_model == "plip":
        from clip import clip
        from transformers import CLIPModel
        _, preprocess = clip.load('ViT-B/16', device=device)
        clip_model = CLIPModel.from_pretrained("vinid/plip").to(device)
        model = clip_model
    elif vlm_model == "quilt1m":
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        clip_model = clip_model.to(device)
        model = clip_model.visual

    model.eval()
    
    l, _ = df_seg_fp.shape
    i = 0
    batch_size = args.batch_size
    
    for row in df_seg_fp.itertuples(index=True, name='Pandas'):
        seg_fp = row.seg_fp  # Now using the tile file directly
        
        wsi_name = seg_fp.split("/")[-1].replace(".h5", "")
        dataset_name = args.label_fp.split("/")[-2].lower()
        
        save_rp_ = os.path.join(args.save_rp, f"{dataset_name}_{args.vlm_model}_{args.base_mag}x_{args.base_patch_size}")
        if not os.path.exists(save_rp_):
            os.makedirs(save_rp_)
        
        out_fp = os.path.join(save_rp_, f'{wsi_name}.pkl')
        
        if os.path.exists(out_fp):
            continue
        
        try:
            data = {}
            print(f"Processing: {seg_fp}")
            
            dataset = TOPDataset(seg_fp, preprocess)
            
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
            
            output = []
            for image in data_loader:
                input = image.to(device)
                with torch.no_grad():
                    if vlm_model == "plip":
                        out = model.visual_projection(model.vision_model(input).pooler_output)
                    elif vlm_model == "clip":
                        out = model(input.type(clip_model.dtype))
                    else:
                        out = model(input)
                
                output.append(out.cpu().detach().numpy())
            
            data["data"] = np.concatenate(output, axis=0)
            
            with open(out_fp, 'wb') as file:
                pickle.dump(data, file)
                
        except Exception as e:
            print(f"Error processing {wsi_name}: {e}")
        
        i += 1    
        print(f"{vlm_model} complete: {i}/{l}")
