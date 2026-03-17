import torch
from pathlib import Path
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Average checkpoint weights')
parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing training checkpoints')
parser.add_argument('--start', type=int, required=True, help='Start step for averaging')
parser.add_argument('--end', type=int, required=True, help='End step for averaging')
parser.add_argument('--step', type=int, required=True, help='Step interval')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for averaged checkpoint')
args = parser.parse_args()

# Generate steps
steps = list(range(args.start, args.end + 1, args.step))

# Get checkpoint paths
ckpt_dir = Path(args.ckpt_dir)
ckpt_paths = [ckpt_dir / f"training_{step}" / "teacher_checkpoint.pth" for step in steps]

# Load checkpoints
state_dicts = [torch.load(str(p), map_location='cpu') for p in ckpt_paths if p.exists()]
teacher_dicts = [sd["teacher"] for sd in state_dicts]

# Average weights
averaged_state_dict = {}
for key in teacher_dicts[0].keys():
    averaged_state_dict[key] = sum(td[key] for td in teacher_dicts) / len(teacher_dicts)

# Save
output_dict = {'teacher': averaged_state_dict}
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(output_dict, output_dir / "teacher_checkpoint.pth")
print(f"Saved averaged checkpoint successfully")